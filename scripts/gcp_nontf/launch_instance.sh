#!/bin/bash
export PROJECT="nnfornlp"
export IMAGE_FAMILY="pytorch-latest-cu92"
export ZONE="us-east1-b"
export INSTANCE_NAME="experiment"
export INSTANCE_TYPE="n1-standard-4"
gcloud compute instances create $INSTANCE_NAME \
        --zone=$ZONE \
        --project=$PROJECT \
        --image-family=$IMAGE_FAMILY \
        --image-project=deeplearning-platform-release \
        --maintenance-policy=TERMINATE \
        --accelerator="type=nvidia-tesla-v100,count=1" \
        --machine-type=$INSTANCE_TYPE \
        --metadata-from-file startup-script=./setup.sh \
        --preemptible \
        --boot-disk-size=120GB \
        --metadata="install-nvidia-driver=True"
