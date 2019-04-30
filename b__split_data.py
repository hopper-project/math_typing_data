import re
import os
import sys
import json
import codecs
import numpy as np
from collections import defaultdict, OrderedDict


file_path = "./task_data.json"
with codecs.open(file_path, 'r', encoding="utf8") as in_f:
  data_raw = json.load(in_f, object_pairs_hook=OrderedDict)
  data = []    # list of data points
  for arxiv_id in data_raw:
    for d_point in data_raw[arxiv_id]:
      d_point["arxiv_id"] = arxiv_id
      data.append(d_point)


def compute_statistics(data):
  class_dict = defaultdict(int)
  print "#data points", len(data)
  for d_point in data:
    type = d_point["type"]
    class_dict[type] += 1
  for type in class_dict:
    sys.stdout.write("%s: %d  " % (type, class_dict[type]))
    print
    sys.stdout.flush()


def split_data(data):
  num_points = len(data)
  np.random.seed(123)
  permutation = list(np.random.choice(range(num_points), num_points, replace = False))
  data = list(np.array(data)[permutation])

  num_train = int(num_points * 0.8)
  num_valid = int(num_points * 0.1)
  num_test = num_points - num_train - num_valid
  train_data = data[:num_train]
  valid_data = data[num_train: num_train + num_valid]
  test_data = data[-num_test:]

  print "train: %d, dev: %d, test: %d" % (len(train_data), len(valid_data), len(test_data))
  with codecs.open("train_data.json","w",encoding="utf8") as json_f:
    json.dump(train_data, json_f, indent=2)
  with codecs.open("valid_data.json","w",encoding="utf8") as json_f:
    json.dump(valid_data, json_f, indent=2)
  with codecs.open("test_data.json","w",encoding="utf8") as json_f:
    json.dump(test_data, json_f, indent=2)


compute_statistics(data)
split_data(data)
