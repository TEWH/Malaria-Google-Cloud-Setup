# TEWH-Malaria-Google-Cloud-Setup


## Google Cloud Configuration


### Initial Steps

- Make a [Google Cloud account](https://cloud.google.com) and sign up for the free $300 credit
- Create a new project
- Navigate to "Compute Engine" -> [VM instances](https://console.cloud.google.com/compute/instances) in the left sidebar


### Create a New Virtual Machine Instance

- Click the blue "Create" button near the middle of the site
- Choose a suitable "Region" and "Zone" that allows for the following configuration (I used Region `us-central1 (Iowa)` and Zone `us-central1-a`)
- Under "Machine configuration", select the following:
    - Machine family: `General-purpose`
    - Series: `N1`
    - Machine type: `n1-standard-1`
    - Click the blue "CPU platform and GPU" dropdown
    - Click "Add GPU" and select `NVIDIA Tesla V100` for the "GPU type" with `1` "Number of GPUs"
- Under "Boot disk", click the "Change" button
    - Change "Image" to `Deep Learning Image: TensorFlow 1.15.0 m41`
- Under "Firewall", select the `Allow HTTP traffic` checkbox
- Click the blue "Create" button and allow ~2 minutes for the instance to boot up


## Jupyter Notebook Configuration


### SSH into the VM

- Navigate to the [VM instances](https://console.cloud.google.com/compute/instances) dashboard
- Click on "SSH" next to your instance in the Google Cloud Console
- Wait until the terminal window is interactable
- Enter `y` if the following prompt appears: "Tshis VM requires Nvidia drivers to function correctly.   Installation takes ~1 minute. Would you like to install the Nvidia driver? [y/n]"
- `nvidia-smi` to confirm a successful driver installation


### Install Dependencies


<!-- #### Git

- `sudo apt install -y git`


#### NVIDIA Driver

- `sudo apt install -y nvidia-driver-435` to install the NVIDIA driver (takes ~5 minutes)
- `sudo shutdown -r now` to restart and complete installation (takes ~5 minutes)
- Reconnect, then `nvidia-smi` to conirm a successful driver installation


#### Docker

- `curl -fsSL https://get.docker.com | sh` to run the install script
- `sudo usermod -a -G docker $USER` to add yourself to the docker group
- `newgrp docker` to join the docker group without having to re-login -->


#### NVIDIA Container Toolkit

Reference: https://github.com/NVIDIA/nvidia-docker

- `distribution=$(. /etc/os-release;echo $ID$VERSION_ID)`
- `curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -`
- `curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list`
- `sudo apt update && sudo apt install -y nvidia-container-toolkit`
- `sudo systemctl restart docker`
- `docker run --gpus all nvidia/cuda nvidia-smi` to verify a successful installation


#### Docker Compose

- `sudo curl -L "https://github.com/docker/compose/releases/download/1.24.1/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose` to download the executable
- `sudo chmod +x /usr/local/bin/docker-compose` to allow execute permissions
- `sudo ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose`


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
