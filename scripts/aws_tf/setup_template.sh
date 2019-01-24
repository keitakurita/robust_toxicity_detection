#!/bin/bash
export GH_TOKEN=""
export AWS_ACCESS_KEY_ID=""
export AWS_SECRET_ACCESS_KEY=""
export AWS_DEFAULT_REGION="us-east-1"
git clone https://$GH_TOKEN@github.com/keitakurita/NNforNLP_Final.git /home/ubuntu/Project
# install python 3.6
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get -y update
sudo apt install -y python3.6
sudo apt-get install build-essential
sudo apt-get install python3.6-dev

# install pipenv and packages
curl https://raw.githubusercontent.com/kennethreitz/pipenv/master/get-pipenv.py | sudo python3.6
(cd Project; pipenv install)

# install awscli
pip install awscli --upgrade --user
