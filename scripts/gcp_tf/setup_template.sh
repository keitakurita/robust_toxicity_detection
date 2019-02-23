export USER_HOME="/home/gcpuser" && \
    echo 'export GH_TOKEN=""' >> $USER_HOME/.bashrc && \
    echo 'export AWS_ACCESS_KEY_ID=""' >> $USER_HOME/.bashrc && \
    echo 'export AWS_SECRET_ACCESS_KEY=""' >> $USER_HOME/.bashrc && \
    echo 'export AWS_DEFAULT_REGION="us-east-1"' >> $USER_HOME/.bashrc && \
    git clone https://$GH_TOKEN@github.com/keitakurita/NNforNLP_Final.git $USER_HOME/Project && \
    sudo cp /tmp/*.ipynb $USER_HOME/Project/notebooks
