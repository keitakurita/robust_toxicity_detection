#!/usr/bin/env python3
"""
Perform basic text transformation

Replace real URL by 'url' token (because some texts have nothing but urls)
Tokenize using the SpaCy model
Preserve casing
Remove repeated symbols
Obtain FastText embeddings for tokens

NOTE: The process assumes that id is col0, comment text is col1

"""

import os, sys
import string
import argparse
import fastText
from itertools import groupby
import json
from allennlp.data import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import  ArrayField, MetadataField
import csv
from allennlp.data.vocabulary import Vocabulary
from preprocessing_common import *
from basic_processing import *
import spacy

from typing import *
from overrides import overrides
from itertools import chain
import numpy as np


NOISE_CONFIG={
  'prob_example_tokens': 0.75,
  'prob_example_distractors': 0.75,
  'targets': {'prob_token': 0.90, 'probs_noise_type': (0.2,0.4,0.4)},
  'nontargets': {'prob_token': 0.5, 'probs_noise_type': (0.2, 0.4, 0.4)},
  'prob_distractor_first': 0.5
}

class NoisingTokenizer(SpacyTokenizer):

  def __init__(self, spacy_nlp, noise_config, toxic_toks_file, vocab_file):
    super().__init__(spacy_nlp)

    self.tokenizer = SpacyTokenizer(spacy_nlp)
    self.toxic_toks = set(np.loadtxt(toxic_toks_file, dtype=str))

    self.noises = [self.noise_homo, self.noise_perm,self.noise_leven]

    # How often distractors are added at the global level
    # i.e. probability that a record will get noised
    self.pr_ex_tokens = noise_config['prob_example_tokens']
    self.pr_ex_distractors = noise_config['prob_example_distractors']

    # Probability that a given targeted token will be modified
    self.pr_tok_targets = noise_config['targets']['prob_token']

    # Probabilities of noise types for targets, 3-ple
    # there are the following types: hypoglyphs, permute, levenshtein
    self.pr_noise_targets = noise_config['targets']['probs_noise_type']

    # Probability that a given non-targeted token will be modified
    self.pr_tok_nontargets = noise_config['nontargets']['prob_token']

    # Probabilities of noise types for non-targets, 3-ple
    # there are the following types: homoglyphs, permute, levenshtein
    self.pr_noise_nontargets = noise_config['nontargets']['probs_noise_type']

    # Probability of distractor noise to be applied first
    self.pr_distractors_first = noise_config['prob_distractor_first']

    self.vocab = np.loadtxt(vocab_file, dtype=str)
    self.leven_index = create_word_index(self.vocab)


  def noise_homo(self, tok, prob=0.2):
    chars = list(tok)
    qty = len(chars)
    draws = np.random.binomial(1, prob, size=qty)
    res = []
    for i in range(qty):
      c = chars[i]
      if draws[i] and c in HOMO_SUBS:
        res.append(HOMO_SUBS[c])
      else:
        res.append(c)
    return ''.join(res)


  def noise_perm(self, tok, max_perm_len=3):
    qty = len(tok)
    if qty < 3:
      return tok
    tok_lst = []
    for k in range(1, qty-1, max_perm_len):
      tmp = list(tok[k:min(qty-1,k+max_perm_len)])
      np.random.shuffle(tmp)
      tok_lst.extend(tmp)
    return tok[0] + ''.join(tok_lst) + tok[-1]

  def noise_leven(self, tok, K=50, max_norm_leven=0.75):
    qres = query_index(self.leven_index, K, [tok])
    if len(qres) == 0:
      return tok
    nids, dists = qres[0]
    cands = []
    cands_dists = []
    for i in range(len(nids)):
      cand_word = self.vocab[nids[i]]
      cand_len = len(cand_word)
      if dists[i] <= min(cand_len, len(tok)) * (1-max_norm_leven) and dists[i] > 0:
        cands.append(cand_word)
        #cands_dists.append(dists[i])
    # for k in range(len(cands)):
    #   print(cands[k], cands_dists[k])
    if not cands:
      return tok
    return np.random.choice(cands)

  def add_distractor(self, toks):

    distr = []

    qty = len(toks)
    start = 0

    while start < qty:

      end = start + 1

      if not toks[start] in self.toxic_toks:

        while end < qty and (not toks[end] in self.toxic_toks):
          end += 1

        if end - start > len(distr):
          distr = toks[start:end]

      start = end

    return toks + distr

  def apply_tok_noise(self, toks):

    res = []

    for tok in toks:

      repl_tox_flag = tok in self.toxic_toks and (np.random.binomial(1, self.pr_tok_targets) > 0)
      repl_ntox_flag = tok not in self.toxic_toks and (np.random.binomial(1, self.pr_tok_nontargets) > 0)

      if repl_tox_flag:

        res.append(self.noises[np.argmax(np.random.multinomial(1, self.pr_noise_targets))](tok))

      if repl_ntox_flag:

        res.append(self.noises[np.argmax(np.random.multinomial(1, self.pr_noise_nontargets))](tok))

      else:

        res.append(tok)

    return res

  @overrides
  def __call__(self, text):

    add_blurb = np.random.binomial(1, self.pr_ex_distractors)
    noise_tokens = np.random.binomial(1, self.pr_ex_tokens)
    add_blurb_first = np.random.binomial(1, self.pr_distractors_first)

    raw_toks = self.tokenizer(text)

    if add_blurb==1 and noise_tokens==1:

      if add_blurb_first==1:
        return self.apply_tok_noise(self.add_distractor(raw_toks))

      else:
        return self.add_distractor(self.apply_tok_noise(raw_toks))

    elif add_blurb==1 and noise_tokens==0:
      return self.add_distractor(raw_toks)

    elif add_blurb==0 and noise_tokens==1:
      return self.apply_tok_noise(raw_toks)

    return raw_toks

def main(argv):

  parser = argparse.ArgumentParser(description='Basic Processing')
  parser.add_argument('--datapath', type=str,
                      required=False,
                      default = '',
                      help = '../data/jigsaw/')
  parser.add_argument('--rawtrain', type=str,
                      required = True,
                      default = 'train.csv',
                      help = 'Raw train file')
  parser.add_argument('--rawtest', type=str,
                      required = True,
                      default = 'test_proced.csv',
                      help = 'Raw test file')
  parser.add_argument('--ftmatname', type=str,
                      required = False,
                      default = 'ft_noised_toks',
                      help = 'Output train file')
  parser.add_argument('--vocabname', type=str,
                      required = False,
                      default = 'voc_noised_toks',
                      help = 'Output vocabulary file')
  parser.add_argument('--proctrain_pref', type=str,
                      required = False,
                      default = 'train_noised',
                      help = 'Output train file')
  parser.add_argument('--proctest_pref', type=str,
                      required = False,
                      default = 'test_noised',
                      help = 'Summary test file')
  parser.add_argument('--ftmodelpath', type=str,
                      required = False,
                      default = 'wiki.en.bin',
                      help = 'Summary test file')
  parser.add_argument('--toxtarget', type=str,
                      required = False,
                      default = 'toxic_targets.txt',
                      help = 'File with toxic target tokens')
  parser.add_argument('--basicvocab', type=str,
                      required = False,
                      default = 'voc_basic_toks/tokens.txt',
                      help = 'File with basic vocabulary')

  args = parser.parse_args(argv)
  print(args)

  set_seed(31416)

  spacy_nlp = spacy.load(SPACY_MODEL_TYPE, disable=['parser', 'ner', 'pos'])

  noise_tok_obj = NoisingTokenizer(spacy_nlp, NOISE_CONFIG,
                                    os.path.join(args.datapath, args.toxtarget),
                                    os.path.join(args.datapath, args.basicvocab)
                                    )

  spacy_ds = get_spacy_vocab_instances(spacy_nlp)
  vocab = Vocabulary.from_instances(spacy_ds)
  print('Spacy vocabulary:', vocab)

  token_indexer = SingleIdTokenIndexer(
    lowercase_tokens=True,
  )

  transformer = JigsawDatasetTransformer(
    TokenTransfomer(noise_tok_obj.spacy_nlp),
    tokenizer=lambda x: noise_tok_obj(x),
    token_indexers={"tokens": token_indexer}
  )

  train_ds = transformer.read(os.path.join(args.datapath,args.rawtrain))
  test_ds = transformer.read(os.path.join(args.datapath,args.rawtest))
  full_ds = list(train_ds) + list(test_ds)

  transformer.save_to_file(train_ds, os.path.join(args.datapath, args.proctrain_pref+".jsonl"))
  transformer.save_to_file(test_ds, os.path.join(args.datapath, args.proctest_pref+".jsonl"))

  transformer.save_to_csv(train_ds,
                          os.path.join(args.datapath, args.proctrain_pref+".csv"),
                          transformer.headers)

  transformer.save_to_csv(test_ds,
                          os.path.join(args.datapath, args.proctest_pref+".csv"),
                          transformer.headers)

  # Need to re-init spacy from scratch
  spacy_ds = get_spacy_vocab_instances(spacy_nlp)
  vocab = Vocabulary.from_instances(chain(spacy_ds, spacy_ds, full_ds))
  vocab.save_to_files(os.path.join(args.datapath, args.vocabname))

  ft_model = fastText.load_model(args.ftmodelpath)

  with open(os.path.join(args.datapath, args.ftmatname+".txt"),"wt") as f:
    for idx, token in vocab.get_index_to_token_vocabulary().items():
      token = token.strip()
      if token:
        emb = ft_model.get_word_vector(token)
        emb_as_str = " ".join(["%.4f" % x for x in emb])
        f.write(f"{token} {emb_as_str}\n")

if __name__ == '__main__':
  main(sys.argv[1:])
