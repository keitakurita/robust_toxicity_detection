aws:
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
	pipenv install

gcp:
	# no need for pipenv
	/opt/anaconda3/bin/pip install -r requirements_gcp.txt
