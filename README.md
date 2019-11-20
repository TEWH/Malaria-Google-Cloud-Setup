# TEWH-Malaria-Google-Cloud-Setup


## Google Cloud Configuration


### Initial Steps

- Make a [Google Cloud account](https://cloud.google.com) and sign up for the free $300 credit
- Create a new project
- Navigate to Compute Engine -> [VM instances](https://console.cloud.google.com/compute/instances) in the left sidebar


## Jupyter Notebook Configuration


### SSH into the VM

- Click on "SSH" next to your instance in the Google Cloud Console


### Install Dependencies


#### Git

- `sudo apt install git`


#### Docker

- `curl -fsSL https://get.docker.com | sh` to run the install script
- `sudo usermod -a -G docker $USER`


#### Docker Compose

- `sudo curl -L "https://github.com/docker/compose/releases/download/1.24.1/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose`
- `sudo chmod +x /usr/local/bin/docker-compose`


### Allow Write Permissions to /srv Directory

- `sudo chmod -R a+w /srv`


### Clone the Repository as /srv/jupyter/

- `cd /srv`
- `git clone https://github.com/TEWH/Malaria-Google-Cloud-Setup.git jupyter`


## Start Jupyter Notebook


### Start the Docker Container

- `cd /srv/jupyter`
- `docker-compose up`


## Stop Jupyter Notebook


### Stop the Docker Container

- `cd /srv/jupter`
- `docker-compose down`


## Pull Updates from GitHub Repository


### Pull the Repository

- `cd /srv/jupter`
- `git pull`


### Rebuild the Docker Image

- `docker-compose build`
- `docker-compose up`
