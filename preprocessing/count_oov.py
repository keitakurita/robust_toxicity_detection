#!/usr/bin/env python
import sys, json

import spacy

from itertools import groupby

from pytorch_pretrained_bert.tokenization import BertTokenizer

nlp = spacy.load("en_core_web_md", disable=['parser', 'ner', 'pos'])
spacy_vocab = set([t.text.lower() for t in nlp.vocab])

bert_vocab = set()
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
for tok, tok_id in bert_tokenizer.vocab.items():
  bert_vocab.add(tok.lower())

inp='../data/jigsaw/train_basic.jsonl'

print(len(bert_vocab), len(spacy_vocab))

oov_spacy = 0
oov_bert = 0
oov_lines_spacy = 0
oov_lines_bert = 0

def remove_extra_chars(s, max_qty=2):
  res = [c * min(max_qty, len(list(group_iter))) for c, group_iter in groupby(s)]
  return ''.join(res)

with open(inp) as f:
  for line in f:
    if line.strip():
      obj = json.loads(line)
      flag_spacy = 0
      flag_bert = 0
      for tok in obj["tokens"]:
        tok = tok.lower()
        if not tok in spacy_vocab:
          oov_spacy += 1
          flag_spacy = 1

        if not tok in bert_vocab:
          oov_bert += 1
          flag_bert = 1

      oov_lines_spacy += flag_spacy
      oov_lines_bert += flag_bert



print(oov_bert, oov_spacy)
print(oov_lines_bert, oov_lines_spacy)

