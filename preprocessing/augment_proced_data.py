#!/usr/bin/env python3
"""
Read data JSONL one line at a time
Apply TBD function
Augment JSON
Write enriched JSONL to another file
"""

import os, sys

sys.path.append('.')

import argparse
import json
import numpy as np
import time
import spacy
from contextlib import contextmanager
from allennlp.common import Tqdm
from typing import *
from preprocessing.preprocessing_common import *

BATCH_SIZE = 5

DictList = List[Dict]


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


def test_aug(dat):
  dat["newelem"] = "TEST"
  return dat


def test_aug_batched(datList: DictList) -> DictList:

  for dat in datList:
    dat["newelem"] = "TEST"

  return datList


def main(argv):

  parser = argparse.ArgumentParser(description='Data augmentation workflow')
  parser.add_argument('--datapath', type=str,
                      required=False,
                      default = '',
                      help = '../data/jigsaw/')
  parser.add_argument('--infile', type=str,
                      required = True,
                      default = 'train_basic.jsonl',
                      help = 'Input file that underwent basic processing')
  parser.add_argument('--outfile', type=str,
                      required = True,
                      default = 'test_augmented.jsonl',
                      help = 'Augmented output file')
  parser.add_argument('--ftmatname', type=str,
                      required = True,
                      default = 'ft_basic_toks.txt',
                      help = 'fastText embeddings file')

  args = parser.parse_args(argv)
  print(args)

  infile = open(os.path.join(args.datapath,args.infile),"r")
  outfile = open(os.path.join(args.datapath,args.outfile),"w")

  spacy_nlp = spacy.load("en_core_web_sm",
                         disable=['parser', 'ner', 'pos'])

  spacy_word_map = get_canon_case_map(spacy_nlp)

  with timer("Loading embeddings"):
    word_arr, embed_arr = read_embeds_and_words_subset(os.path.join(args.datapath,args.ftmatname), spacy_word_map)
    print('Read %d spacy words from the fasttext-dictionary file' % len(word_arr))

  embed_index = create_embeds_index(embed_arr)

  word_index = create_word_index(word_arr)

  with timer("Augmenting data"):

    dat_list = []

    for _, line in Tqdm.tqdm(enumerate(infile)):
      dat_list.append(json.loads(line))

    infile.close()

    data_qty = len(dat_list)

    for batch_s in Tqdm.tqdm(range(0, data_qty, BATCH_SIZE)):

        batch_e = min(data_qty, batch_s + BATCH_SIZE)
        batch_data = dat_list[batch_s:batch_e]

        test_aug_batched(batch_data)

        for dat in batch_data:
          datStr = json.dumps(dat)
          outfile.write(datStr + '\n')

  outfile.close()


if __name__ == '__main__':
  main(sys.argv[1:])

