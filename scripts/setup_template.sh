#!/bin/bash
export GH_TOKEN=""
git clone https://$GH_TOKEN@github.com/CloudComputingTeamProject/Nephologists-F18.git /home/ubuntu/Project
(cd ..; pipenv install)
