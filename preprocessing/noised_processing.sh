#!/usr/bin/env bash

# Tests
#python -u ./noised_processing.py --datapath ~/NNforNLP_Final/data/jigsaw --rawtrain small_train.csv --rawtest small_test_proced.csv --ftmatname ft_noised_toks --vocabname voc_noised_toks --proctrain_pref train_noised --proctest_pref test_noised --ftmodelpath ~/NNforNLP_Final/data/jigsaw/wiki.en.bin --toxtarget toxic_targets.txt --basicvocab voc_basic_toks/tokens.txt 2>&1|tee log_basic

# Generate the test set with policy

#NOISE_CONFIG={
#  'prob_example_tokens': 0.99,
#  'prob_example_distractors': 0.01,
#  'targets': {'prob_token': 0.99, 'probs_noise_type': (0.4, 0.4, 0.2)},
#  'nontargets': {'prob_token': 0.01, 'probs_noise_type': (0.4, 0.4, 0.2)},
#  'prob_distractor_first': 0.5,
#   'number_of_distractors' : 2
#}

#python -u ./noised_processing.py --datapath ~/NNforNLP_Final/data/jigsaw --rawtrain small_train.csv --rawtest test_proced.csv --ftmatname ft_noised_toks --vocabname voc_noised_toks --proctrain_pref train_noised --proctest_pref test_noised --ftmodelpath ~/NNforNLP_Final/data/jigsaw/wiki.en.bin --toxtarget toxic_targets.txt --basicvocab voc_basic_toks/tokens.txt 2>&1|tee log_test_noise

# Generate the test set with policy

#NOISE_CONFIG={
#  'prob_example_tokens': 0.99,
#  'prob_example_distractors': 0.99,
#  'targets': {'prob_token': 0.99, 'probs_noise_type': (0.4, 0.4, 0.2)},
#  'nontargets': {'prob_token': 0.01, 'probs_noise_type': (0.4, 0.4, 0.2)},
#  'prob_distractor_first': 0.5,
#   'number_of_distractors' : 2
#}

#python -u ./noised_processing.py --datapath ~/NNforNLP_Final/data/jigsaw --rawtrain small_train.csv --rawtest test_proced.csv --ftmatname ft_noised_toks --vocabname voc_noised_toks --proctrain_pref train_noised --proctest_pref test_noised_distract --ftmodelpath ~/NNforNLP_Final/data/jigsaw/wiki.en.bin --toxtarget toxic_targets.txt --basicvocab voc_basic_toks/tokens.txt 2>&1|tee log_test_noise


# Generate the test set with policy - 50k toxic terms

#NOISE_CONFIG={
#  'prob_example_tokens': 0.99,
#  'prob_example_distractors': 0.01,
#  'targets': {'prob_token': 0.99, 'probs_noise_type': (0.4, 0.4, 0.2)},
#  'nontargets': {'prob_token': 0.01, 'probs_noise_type': (0.4, 0.4, 0.2)},
#  'prob_distractor_first': 0.5,
#   'number_of_distractors' : 2
#}

#python -u ./noised_processing.py --datapath ~/NNforNLP_Final/data/jigsaw --rawtrain small_train.csv --rawtest test_proced.csv --ftmatname ft_noised_toks --vocabname voc_noised_toks --proctrain_pref train_noised --proctest_pref test_noised_large --ftmodelpath ~/NNforNLP_Final/data/jigsaw/wiki.en.bin --toxtarget toxic_targets.txt --basicvocab voc_basic_toks/tokens.txt 2>&1|tee log_test_noise

# Generate the test set with policy - 50k toxic terms

#NOISE_CONFIG={
#  'prob_example_tokens': 0.99,
#  'prob_example_distractors': 0.99,
#  'targets': {'prob_token': 0.99, 'probs_noise_type': (0.4, 0.4, 0.2)},
#  'nontargets': {'prob_token': 0.01, 'probs_noise_type': (0.4, 0.4, 0.2)},
#  'prob_distractor_first': 0.5,
#   'number_of_distractors' : 2
#}

#python -u ./noised_processing.py --datapath ~/NNforNLP_Final/data/jigsaw --rawtrain small_train.csv --rawtest test_proced.csv --ftmatname ft_noised_toks --vocabname voc_noised_toks --proctrain_pref train_noised --proctest_pref test_noised_distract_large --ftmodelpath ~/NNforNLP_Final/data/jigsaw/wiki.en.bin --toxtarget toxic_targets.txt --basicvocab voc_basic_toks/tokens.txt 2>&1|tee log_test_noise


# Generate the TRAIN set with policy - 50k toxic terms and symmetric hard noise

#NOISE_CONFIG={
#  'prob_example_tokens': 0.99,
#  'prob_example_distractors': 0.99,
#  'targets': {'prob_token': 0.99, 'probs_noise_type': (0.4, 0.4, 0.2)},
#  'nontargets': {'prob_token': 0.99, 'probs_noise_type': (0.4, 0.4, 0.2)},
#  'prob_distractor_first': 0.5,
#   'number_of_distractors' : 2
#}

#python -u ./noised_processing.py --datapath ~/NNforNLP_Final/data/jigsaw --rawtrain train.csv --rawtest test_proced.csv --ftmatname ft_symmetric_hard_noised_toks --vocabname voc_symmetric_hard_noised_toks --proctrain_pref train_symmetric_hard_noised --proctest_pref test_symmetric_hard_noised --ftmodelpath ~/NNforNLP_Final/data/jigsaw/wiki.en.bin --toxtarget toxic_targets.txt --basicvocab voc_basic_toks/tokens.txt 2>&1|tee log_test_noise

# Generate the TRAIN set with policy - 50k toxic terms and larger adversarial noise

#NOISE_CONFIG={
#  'prob_example_tokens': 0.99,
#  'prob_example_distractors': 0.99,
#  'targets': {'prob_token': 0.99, 'probs_noise_type': (0.4, 0.4, 0.2)},
#  'nontargets': {'prob_token': 0.5, 'probs_noise_type': (0.2, 0.3, 0.5)},
#  'prob_distractor_first': 0.5,
#   'number_of_distractors' : 2
#}

#python -u ./noised_processing.py --datapath ~/NNforNLP_Final/data/jigsaw --rawtrain train.csv --rawtest test_proced.csv --ftmatname ft_asymmetric_noised_toks --vocabname voc_asymmetric_noised_toks --proctrain_pref train_asymmetric_noised --proctest_pref test_asymmetric_noised --ftmodelpath ~/NNforNLP_Final/data/jigsaw/wiki.en.bin --toxtarget toxic_targets.txt --basicvocab voc_basic_toks/tokens.txt 2>&1|tee log_test_noise


# Generate the TRAIN set with policy - 50k toxic terms and symmetric natural noise

#NOISE_CONFIG={
#  'prob_example_tokens': 0.99,
#  'prob_example_distractors': 0.99,
#  'targets': {'prob_token': 0.5, 'probs_noise_type': (0.2, 0.3, 0.5)},
#  'nontargets': {'prob_token': 0.5, 'probs_noise_type': (0.2, 0.3, 0.5)},
#  'prob_distractor_first': 0.5,
#   'number_of_distractors' : 2
#}

#python -u ./noised_processing.py --datapath ~/NNforNLP_Final/data/jigsaw --rawtrain train.csv --rawtest test_proced.csv --ftmatname ft_symmetric_natural_noised_toks --vocabname voc_symmetric_natural_noised_toks --proctrain_pref train_symmetric_natural_noised --proctest_pref test_symmetric_natural_noised --ftmodelpath ~/NNforNLP_Final/data/jigsaw/wiki.en.bin --toxtarget toxic_targets.txt --basicvocab voc_basic_toks/tokens.txt 2>&1|tee log_test_noise

# Generate the TRAIN set with policy - 50k toxic terms and asymmetric hard noise only, with distractor

#NOISE_CONFIG={
#  'prob_example_tokens': 0.99,
#  'prob_example_distractors': 0.99,
#  'targets': {'prob_token': 0.99, 'probs_noise_type': (0.4, 0.4, 0.2)},
#  'nontargets': {'prob_token': 0.01, 'probs_noise_type': (0.4, 0.4, 0.2)},
#  'prob_distractor_first': 0.5,
#   'number_of_distractors' : 2
#}

#python -u ./noised_processing.py --datapath ~/NNforNLP_Final/data/jigsaw --rawtrain train.csv --rawtest test_proced.csv --ftmatname ft_noised_distract_large_toks --vocabname voc_noised_distract_large_toks --proctrain_pref train_noised_distract_large --proctest_pref test_noised_distract_large --ftmodelpath ~/NNforNLP_Final/data/jigsaw/wiki.en.bin --toxtarget toxic_targets.txt --basicvocab voc_basic_toks/tokens.txt 2>&1|tee log_test_noise

# Generate the TRAIN set with policy - 50k toxic terms and asymmetric hard noise only, no distractors

#NOISE_CONFIG={
#  'prob_example_tokens': 0.99,
#  'prob_example_distractors': 0.01,
#  'targets': {'prob_token': 0.99, 'probs_noise_type': (0.4, 0.4, 0.2)},
#  'nontargets': {'prob_token': 0.01, 'probs_noise_type': (0.4, 0.4, 0.2)},
#  'prob_distractor_first': 0.5,
#   'number_of_distractors' : 2
#}

python -u ./noised_processing.py --datapath ~/NNforNLP_Final/data/jigsaw --rawtrain train.csv --rawtest test_proced.csv --ftmatname ft_noised_large_toks --vocabname voc_noised_large_toks --proctrain_pref train_noised_large --proctest_pref test_noised_large --ftmodelpath ~/NNforNLP_Final/data/jigsaw/wiki.en.bin --toxtarget toxic_targets.txt --basicvocab voc_basic_toks/tokens.txt 2>&1|tee log_test_noise


