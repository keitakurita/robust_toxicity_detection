#!/usr/bin/env bash

#time python -u augment_proced_data.py --datapath ~/NNforNLP_Final/data/jigsaw/ --infile test_basic.jsonl --outfile test_basic_aug.jsonl --ftmatname ft_basic_toks.txt 2>&1|tee log_test_aug

time python -u augment_proced_data.py --datapath ~/NNforNLP_Final/data/jigsaw/ --infile test_noised_distract_large.jsonl --outfile test_noised_distract_large_aug.jsonl --ftmatname ft_noised_distract_large_toks.txt 2>&1|tee log_test_noised_aug

time python -u augment_proced_data.py --datapath ~/NNforNLP_Final/data/jigsaw/ --infile train_basic.jsonl --outfile train_basic_aug.jsonl --ftmatname ft_basic_toks.txt 2>&1|tee log_train_aug