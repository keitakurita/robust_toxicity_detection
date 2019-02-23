#!/bin/bash
set -e
echo 'export GH_TOKEN=""' >> $HOME/.bashrc
echo 'export AWS_ACCESS_KEY_ID=""' >> $HOME/.bashrc
echo 'export AWS_SECRET_ACCESS_KEY=""' >> $HOME/.bashrc
echo 'export AWS_DEFAULT_REGION="us-east-1"' >> $HOME/.bashrc
git clone https://$GH_TOKEN@github.com/keitakurita/NNforNLP_Final.git /home/ubuntu/Project

# copy uploaded notebook to notebook directory
sudo cp /tmp/*.ipynb $HOME/Project/notebooks

echo "Experiment deployed"
