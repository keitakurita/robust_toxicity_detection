#!/usr/bin/env python3
"""
Identifies target words for noising
using new (large) jigsaw dataset released in Spring 2019
confirms that there are no leaks from the older dataset
TODO: for now does not confirm leaks, assumes that thise were removed in train_new_jigsaw.csv
"""

import os, sys
sys.path.append('.')

import argparse
import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from preprocessing_common import *

import spacy
nlp = spacy.load('en_core_web_sm')

DEBUG = True
N_JOBS = 6
MIN_DF = 50
NUM_TERMS = 10000

def tok(s): return [tok.text for tok in nlp.tokenizer(s)]


def MostIndicativeN(vectorizer, clf, N):

  feature_names = vectorizer.get_feature_names()
  coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))

  topClass1 = coefs_with_fns[:N]
  topClass2 = coefs_with_fns[:-(N + 1):-1]

  print("Class 1 best: ")
  class1_toks = []
  for feat in topClass1:
    print(feat)
    class1_toks.append(feat[1])

  print("Class 2 best: ")
  class2_toks = []
  for feat in topClass2:
    print(feat)
    class2_toks.append(feat[1])

  return class1_toks, class2_toks

def remove_leaks(source, train, test):
  source_id = set(source["comment_text"])
  train_id = set(train["comment_text"])
  test_id = set(test["comment_text"])
  new_id = source_id - train_id - test_id
  return source[(source["comment_text"].isin(new_id))]

def transform_label(row):
  if row['target'] > 0.5:
    return 1
  else:
    return 0

def main(argv):

  parser = argparse.ArgumentParser(description='Data loader')
  parser.add_argument('--datapath', type=str,
                      required=False,
                      default = '',
                      help = '../data/jigsaw/')
  parser.add_argument('--donorfile', type=str,
                      required = True,
                      default = 'train_new_jigsaw.csv',
                      help = 'New jigsaw file to source noise targets')
  parser.add_argument('--trainfile', type=str,
                      required = True,
                      default = 'train.csv',
                      help = 'Old jigsaw train file used in core modeling')
  parser.add_argument('--testfile', type=str,
                      required = True,
                      default = 'test_proced.csv',
                      help = 'Old jigsaw test file used in modeling')
  parser.add_argument('--numterms', type=int,
                      required = True,
                      default = NUM_TERMS,
                      help = 'Number of terms for targeting')


  args = parser.parse_args(argv)
  print(args)

  donor_file = pd.read_csv(os.path.join(args.datapath,args.donorfile))
  train_file = pd.read_csv(os.path.join(args.datapath,args.trainfile))
  test_file = pd.read_csv(os.path.join(args.datapath,args.testfile))
  source_file = remove_leaks(donor_file,train_file,test_file)

  if DEBUG:
    print("Original file",len(donor_file))
    print("No leaks file", len(source_file))

  vectorizer = CountVectorizer(tokenizer=tok, ngram_range=(1, 1),min_df=MIN_DF)
  clf = LogisticRegression(n_jobs=N_JOBS)
  pipe = Pipeline([('vectorizer', vectorizer), ('clf', clf)])

  X = source_file['comment_text'].tolist()
  Y = np.array(source_file.apply(lambda row: transform_label(row) , axis=1))

  pipe.fit(X, Y)

  class1_toks, class2_toks = MostIndicativeN(vectorizer, clf, args.numterms)

  np.savetxt(os.path.join(args.datapath,"toxic_targets.txt"),class2_toks,fmt='%s')
  np.savetxt(os.path.join(args.datapath,"distractor_targets.txt"),class1_toks,fmt='%s')

if __name__ == '__main__':
  main(sys.argv[1:])
