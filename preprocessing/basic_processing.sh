#!/usr/bin/env bash

#python -u ./basic_processing.py --datapath ~/NNforNLP_Final/data/jigsaw --rawtrain train.csv --rawtest test_proced.csv --ftmatname ft_basic_toks --vocabname voc_basic_toks --proctrain train_basic.jsonl --proctest test_basic.jsonl --ftmodelpath ~/NNforNLP_Final/data/jigsaw/wiki.en.bin 2>&1|tee log_basic

#jigsaw_noised_old

#for i in homoglyph insert remove repeat swap; do

#    python -u ./basic_processing.py --datapath ~/NNforNLP_Final/data/jigsaw_noised_old --rawtrain train_noised_${i}.csv --rawtest test_noised_${i}.csv --ftmatname ft_noised_${i}_toks --vocabname voc_noised_${i}_toks --proctrain train_noised_${i}.jsonl --proctest test_noised_${i}.jsonl --ftmodelpath ~/NNforNLP_Final/data/jigsaw/wiki.en.bin 2>&1|tee log_noised_old_${i}

#done


#jigsaw_noised_new

for i in combined; do

    python -u ./basic_processing.py --datapath ~/NNforNLP_Final/data/jigsaw_noised_new --rawtrain train_noise_${i}.csv --rawtest test_noise_${i}.csv --ftmatname ft_noised_${i}_toks --vocabname voc_noised_${i}_toks --proctrain train_noised_${i}.jsonl --proctest test_noised_${i}.jsonl --ftmodelpath ~/NNforNLP_Final/data/jigsaw/wiki.en.bin 2>&1|tee log_noised_old_${i}

done


