import re
import os
import json
import codecs
import string
from collections import defaultdict, OrderedDict

# import nltk

# for parallel
from joblib import Parallel, delayed
import multiprocessing

a_z = list(string.ascii_lowercase)
A_Z = list(string.ascii_uppercase)
greek = "\\alpha \\beta \\gamma \\Gamma \\delta \\Delta \\epsilon \\zeta \\eta \\theta \\Theta \\iota \\kappa \\lambda \\Lambda \\mu \\nu \\omicron \\pi \\Pi \\rho \\sigma \\Sigma \\tau \\upsilon \\Upsilon \\phi \\Phi \\chi \\psi \\Psi \\omega \\Omega".split()
var_vocab_toks = a_z + A_Z + greek

def is_display_eq(word):
  return ((word.startswith("EQDS") or word.startswith("EQDM")) and word.endswith("Q"))

def is_inline_eq(word):
  return (word.startswith("EQIX") and word.endswith("Q"))

def is_eq(word):
  return ((word.startswith("EQDS") or word.startswith("EQDM") or word.startswith("EQIX")) and word.endswith("Q"))


def load_eq_dict(arxiv_id, inline):
  ROOT = "../token_eqs_tsv/"
  month = arxiv_id.split(".")[0]
  fpath = None
  if inline:
    fpath = ROOT + "%s/%s_inline.tsv" % (month, arxiv_id)
  else:
    fpath = ROOT + "%s/%s.tsv" % (month, arxiv_id)
  eq_dict = {}
  with codecs.open(fpath,'r',encoding="utf8") as fh:
    for line in fh:
      linesplit = line.rstrip('\n').split('\t')
      if len(linesplit)==2:
        eq = linesplit[1].split() # list of math tokens
      elif len(linesplit)==3:
        eq = linesplit[2].split() # list of math tokens
      else: continue
      eqid = linesplit[0]
      eq_dict[eqid] = eq
  return eq_dict

def match_def_pattern(sent, type_vocab, eq_dict):
  def should_return(match, type_vocab, eq_dict):
    if match is None: return False
    eq_id = match.group(1)
    type  = match.group(4)
    if type not in type_vocab: return False
    eq = eq_dict[eq_id]
    var = eq[0]
    if var not in var_vocab_toks: return False
    return True

  # sent = sent.lower()
  # match = re.search(r'(EQ(IX)[\d]+?Q) is (a|an) ([\w]+?)(,|\.|\s)', sent)
  # if should_return(match, type_vocab, eq_dict):
  #   return match
  match = re.search(r'(EQ(IX)[\d]+?Q) (?:is|denotes|represents) (a|an|the) ([\w]+?)(,|\.|\s)', sent)  # ?: avoids capture
  if should_return(match, type_vocab, eq_dict):
    return match
  match = re.search(r'[lL]et (EQ(IX)[\d]+?Q) be (a|an|the) ([\w]+?)(,|\.|\s)', sent)
  if should_return(match, type_vocab, eq_dict):
    return match
  return None

def get_gold_labels(sents, type_vocab, eq_dict):
  gold_labels = {}
  for (s_id, sent) in enumerate(sents):
    match = match_def_pattern(sent, type_vocab, eq_dict)
    if match is not None:
      # print "yes!"
      eq_id = match.group(1)
      type  = match.group(4)
      eq = eq_dict[eq_id]
      var = eq[0];  #assert var in var_vocab_toks
      if var not in var_vocab_toks: continue
      if var not in gold_labels:
        gold_labels[var] = {"eq": eq, "var": var, "type": type, "sent": sent, "sent_id": s_id}
  return gold_labels

def is_var_in_eq(eq_toks, var):
  eq = " ".join(eq_toks)
  eq = re.sub(r'\\operatorname[\*]?[\s]?{[\d\D]+?\s[\d\D]+?}', " ", eq)
  eq = re.sub(r'\\mathrm[\*]?[\s]?{[\d\D]+?\s[\d\D]+?}', " ", eq)
  eq = re.sub(r'\\textrm[\*]?[\s]?{[\d\D]+?\s[\d\D]+?}', " ", eq)
  # e.g. remove  \mathrm { e x p }  but include  \mathrm { H }
  eq_toks = eq.split()
  return (var in eq_toks)

def retreive_disp_eq(sents, eq_dict, var):
  #retrive display eq containing a given variable
  ret = []
  for (s_id, sent) in enumerate(sents):
    for word in sent.split():
      if is_display_eq(word):
        if word not in eq_dict:
          continue
        eq_toks = eq_dict[word]
        if eq_toks[0][0] == '$': continue
        if len(eq_toks) < 5 or len(eq_toks) > 50: continue
        if not is_var_in_eq(eq_toks, var): continue
        ret.append(" ".join(eq_toks))
  return ret

def retreive_text(sents, eq_dict, var, gold_sent_id):
  #retrive text + inline_eq containing a given variable
  ret = []
  sents_weq = []  #sents with eq content
  for (s_id, sent) in enumerate(sents):
    sent_weq = []
    for word in sent.split():
      if is_inline_eq(word):
        eq_toks = eq_dict[word]
        sent_weq.append('$'+ " ".join(eq_toks) +'$')
      else:
        sent_weq.append(word)
    sents_weq.append(" ".join(sent_weq))
  for (s_id, sent) in enumerate(sents):
    # if gold_sent_id -1 <= s_id <= gold_sent_id +1: continue
    # if gold_sent_id == s_id: continue
    for word in sent.split():
      if is_inline_eq(word):
        eq_toks = eq_dict[word]
        if len(eq_toks) > 20: continue
        if not is_var_in_eq(eq_toks, var): continue
        # ret.append(" ".join(sents_weq[s_id-1: s_id+2]))
        if gold_sent_id == s_id:
          pass # ret.append("[Gold Sentence] " + sents_weq[s_id])
        else:
          ret.append(sents_weq[s_id])
        break
  return ret

def get_test_data(sents, eq_dict, gold_labels):
  test_data = []
  for var in gold_labels:
    gold_label = gold_labels[var]
    raw  = gold_label["eq"]
    type = gold_label["type"]
    gold_sent_id = gold_label["sent_id"]
    data_point = OrderedDict([("var", var), ("raw", raw), ("type", type),
                  # ("gold_sent_id", gold_sent_id),
                  ("gold_sent", sents[gold_sent_id])])
    found_eqs = retreive_disp_eq(sents, eq_dict, var)
    found_texts = retreive_text(sents, eq_dict, var, gold_sent_id)
    if len(found_eqs) > 0 and len(found_texts) > 0:
      data_point["eqs"] = found_eqs
      data_point["texts"] = found_texts
      test_data.append(data_point)
      # print data_point.keys()
  return test_data


def process_one_paper(i, arxiv_id, paper, eval_tmp_d, type_vocab):
  if i % 1000 == 0: print i
  json_fpath = eval_tmp_d + "/%s.json" % arxiv_id
  if os.path.exists(json_fpath): return

  sents = paper
  if len(sents) < 5 or len(sents) > 1000: return
  eq_dict = load_eq_dict(arxiv_id, inline=True)
  eq_dict.update(load_eq_dict(arxiv_id, inline=False))

  gold_labels = get_gold_labels(sents, type_vocab, eq_dict)
  if len(gold_labels) == 0:
    return

  test_data = get_test_data(sents, eq_dict, gold_labels)
  if len(test_data) == 0:
    return

  with codecs.open(json_fpath,"w",encoding="utf8") as json_f:
    json.dump(test_data, json_f, indent=2)

def process_ngrams(type_vocab, papers):
  # change "real number" => "read_number"
  # up to bigram
  for arxiv_id in papers:
    paper = papers[arxiv_id]
    for (s_id, sent) in enumerate(paper):
      words = sent.split(); s_len = len(words); w_id = 0
      new_words = []
      while w_id < s_len:
        if " ".join(words[w_id: w_id+2]) in type_vocab:
          new_words.append("_".join(words[w_id: w_id+2]))
          w_id += 2
        else:
          new_words.append(words[w_id])
          w_id += 1
      paper[s_id] = " ".join(new_words)
  types = type_vocab.keys()
  for type in types:
    type_split = type.split()
    if len(type_split) > 1:
      type_vocab["_".join(type_split)] = type_vocab.pop(type)


def get_eval_set():
  vocab_PATH = "../2_get_math_type_vocab/save/math_types_john_pick.json"
  data_PATH  = "../1_preprocess/1__tokenized_json/tokenized_full.json"
  # data_PATH  = "./100.json"
  eval_tmp_d = "./eval_data_tmp"
  if not os.path.exists(eval_tmp_d): os.mkdir(eval_tmp_d)
  # if os.path.exists(eval_tmp_d): os.system("rm -r %s" % eval_tmp_d)
  # os.mkdir(eval_tmp_d)
  type_vocab = json.load(codecs.open(vocab_PATH,"r",encoding="utf8"))
  papers = json.load(codecs.open(data_PATH,"r",encoding="utf8"), object_pairs_hook=OrderedDict)
  print "Loaded papers"

  print "preprocess ngrams..."
  process_ngrams(type_vocab, papers)

  print "get task data in parallel..."
  Parallel(n_jobs=100)(delayed(process_one_paper)(i, arxiv_id, paper, eval_tmp_d, type_vocab) for (i,(arxiv_id, paper)) in enumerate(papers.items()))


  eval_set = OrderedDict() #key: arxiv_id,  val: gold (var, type) pair
  for fname in sorted(os.listdir(eval_tmp_d)):
    arxiv_id = fname[:-len(".json")]
    fpath = eval_tmp_d + "/" + fname
    with codecs.open(fpath,"r",encoding="utf8") as json_f:
      gold_point = json.load(json_f, object_pairs_hook=OrderedDict)
    eval_set[arxiv_id] = gold_point

  print "#papers:", len(eval_set)
  print "#data points:", sum([len(eval_set[arxiv_id]) for arxiv_id in eval_set])
  with codecs.open("task_data.json","w",encoding="utf8") as json_f:
    json.dump(eval_set, json_f, indent=2)
  # os.system("rm -r %s" % eval_tmp_d)


# def get_100_json():
#   data_PATH  = "../1_preprocess/1__tokenized_json/tokenized_full.json"
#   papers = json.load(codecs.open(data_PATH,"r",encoding="utf8"), object_pairs_hook=OrderedDict)
#   print "Loaded papers"
#   papers = OrderedDict(papers.items()[:100])
#   json.dump(papers, codecs.open("100.json","w",encoding="utf8"), indent=2)


#### main ####
get_eval_set()
# get_100_json()
