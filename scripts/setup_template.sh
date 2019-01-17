#!/bin/bash
export GH_TOKEN=""
git clone https://$GH_TOKEN@github.com/keitakurita/NNforNLP_Final.git /home/ubuntu/Project
(cd ..; pipenv install)
