#!/bin/bash
set -e
export GH_TOKEN=""
export AWS_ACCESS_KEY_ID=""
export AWS_SECRET_ACCESS_KEY=""
export AWS_DEFAULT_REGION="us-east-1"
git clone https://$GH_TOKEN@github.com/keitakurita/NNforNLP_Final.git /home/ubuntu/Project
# install python 3.6
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get -y update && \
    sudo apt install -y python3.6 && \
    sudo apt-get install -y build-essential && \
    sudo apt-get install -y python3.6-dev

# install pipenv and packages
sudo pip install pipenv

# install awscli
sudo pip install awscli --upgrade

# copy uploaded notebook to notebook directory
sudo cp /tmp/*.ipynb $HOME/Project/notebooks

# run experiment
(cd $HOME/Project; pipenv install)
(cd $HOME/Project/scripts; pipenv run python run_experiment.py --file /tmp/experiment_config.yaml &)
echo "Experiment deployed"
