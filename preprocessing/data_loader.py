#!/usr/bin/env python3
"""
Creates instances from processed JSONL that includes

tokens
word level features
sentence level features
word substitutions if as available
metadata (id, raw text)

"""


import os, sys
sys.path.append('.')
sys.path.append('..')

import argparse
import json

import numpy as np

from allennlp.data import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import ArrayField, MetadataField

from preprocessing_common import *

from typing import *
from overrides import overrides

def dummy_token_extender(toks):
  return toks

reader_registry = {}

def register(name: str):
  def dec(x: Callable):
    reader_registry[name] = x
    return x

  return dec

class OOVTokenSwapper:

  def __init__(self,
               global_weight = 0.5,
               context_weight = 0.25,
               surface_weight = 0.25):
    self.gw = global_weight
    self.cw = context_weight
    self.sw = surface_weight

  def _get_token(self,oov_token, candidates):

    max_score = 0
    swap_token = oov_token
    for c in candidates:
      score =  self.gw * candidates[c][FASTTEXT_SIMIL] + \
               self.cw * candidates[c][BERT_SIMIL]  + \
               self.sw * candidates[c][LEVEN_SIMIL]
      if score > max_score:
        max_score = score
        swap_token = c

    return swap_token

  def swap(self,
           tokens: List[str],
           oov_tokens:  List[Dict] = None) -> List[str]:

     if oov_tokens is None:
       pass

     if self.gw == 0 and self.cw == 0 and self.sw == 0:
       pass

     for oov in oov_tokens:
       swap = self._get_token(oov['oov_token'],oov['cand_scores'])
       if swap is not None:
         pos = oov['sent_pos_oov']
         tokens[pos] = swap

     return tokens

@register("toxicdata")
class ToxicDatasetJSONLReader(DatasetReader):

  def __init__(self,
               token_extender : Callable[[str], List[str]],
               oov_token_swapper: OOVTokenSwapper,
               testing: bool = False,
               token_indexers: Dict[str, TokenIndexer] = None,
               max_seq_len: Optional[int] = MAX_SEQ_LEN,
               label_smoothing_eps = 0.) -> None:
    super().__init__(lazy=False)
    self.token_extender = token_extender
    self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
    self.oov_token_swapper = oov_token_swapper
    self.max_seq_len = max_seq_len
    self.testing = testing
    self.label_smoothing_eps = label_smoothing_eps

  def smooth_labels(self,labels):
    if self.label_smoothing_eps > 0. :
      return (1-self.label_smoothing_eps)*labels + self.label_smoothing_eps/2
    else:
      return labels

  @overrides
  def text_to_instance(self, tokens: List[str],
                       wl_feats: List[float],
                       sl_feats: List[float],
                       labels: List[int],
                       id: str,
                       oov: List[Dict] = None) -> Instance:

    fields = {}

    if oov is None:

      sentence_field = MemoryOptimizedTextField(self.token_extender(tokens), self.token_indexers)

    else:

      sentence_field = MemoryOptimizedTextField(self.token_extender(self.oov_token_swapper.swap(tokens,oov)),
                                                self.token_indexers)

    fields["tokens"] = sentence_field

    fields["word_level_features"] = ArrayField(array=np.array(wl_feats))

    fields["sentence_level_features"] = ArrayField(array=np.array(sl_feats))

    label_field = ArrayField(array=self.smooth_labels(np.array(labels)))
    fields["label"] = label_field


    fields["id"] = MetadataField(id)


    return Instance(fields)

  @overrides
  def _read(self, file_path: str) -> Iterator[Instance]:

    with open(file_path) as f:

      for i, line in enumerate(f):

        data = json.loads(line)

        if "oov" in data:
          oov_data = data["oov"]
        else:
          oov_data = None

        yield self.text_to_instance(
            data["tokens"],
            data["word_level_features"],
            data["sentence_level_features"],
            data["label"],
            data["id"],
            oov_data
            )

        if self.testing and i == 1000: break


def main(argv):

  parser = argparse.ArgumentParser(description='Data loader')
  parser.add_argument('--datapath', type=str,
                      required=False,
                      default = '',
                      help = '../data/jigsaw/')
  parser.add_argument('--infile', type=str,
                      required = True,
                      default = 'train_basic.jsonl',
                      help = 'Input file that underwent processing')


  args = parser.parse_args(argv)
  print(args)

  token_indexer = SingleIdTokenIndexer(
    lowercase_tokens=True,
  )

  reader = ToxicDatasetJSONLReader(
    token_extender = dummy_token_extender,
    oov_token_swapper = OOVTokenSwapper(),
    token_indexers={"tokens": token_indexer}
  )

  ds = reader.read(os.path.join(args.datapath,args.infile))
  print(len(ds))
  print(vars(ds[0]))

if __name__ == '__main__':
  main(sys.argv[1:])