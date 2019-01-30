# 11-747 NN for NLP final project
All code comes here.
Directory layout should be self-explanitory

## Running Experiments

Still ironing out a lot of bugs...

### GCP
1. Run `$ make; terraform init; terraform apply` in scripts/gcp\_tf
2. Login to the vm and install the CUDA driver
3. Clone this repo
4. Run `$ make gcp` at the root of this repo

### AWS
1. Run the launch\_experiment.py script (See help/code for details)
2. The launch will probably fail. Login to the instance.
3. Wait until you can run apt-get (this will take a lot of time, can't AWS make this faster??)
4. Install necessary stuff and run experiment.

### Colab
1. Upload notebook to colab
2. Set `os.environ["IS\_COLAB"] = "True"`
3. Run as normal
