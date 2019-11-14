# TEWH-Malaria-Google-Cloud-Setup

## Google Cloud Configuration

### Initial Steps

- Make a [Google Cloud account](https://cloud.google.com) and sign up for the free $300 credit
- Create a new project
- Navigate to Compute Engine -> [VM instances](https://console.cloud.google.com/compute/instances) in the left sidebar


### Start Preemptible VM Instance

- Reference: https://cloud.google.com/preemptible-vms
- `gcloud compute instances create tewh-malaria --image-project=ml-images --image-family=tf-1-15 --zone us-central1-c --scopes=cloud-platform --preemptible`


### Start Preemptible TPU

- Reference: https://cloud.google.com/tpu/docs/preemptible
- `gcloud compute tpus create tewh-malaria --zone=us-central1-c --network=default --accelerator-type=v2-8 --range=192.168.0.0 --version=1.15 --preemptible`
- `ctpu up --tpu-only --name=tewh-malaria --zone=us-central1-c --preemptible`


### Start VM + TPU Together

- `ctpu up --name=tewh-malaria --zone=us-central1-c --tpu-size=v2-8 --machine-type=n1-standard-1 --disk-size-gb=25 --preemptible`


### Link TPU with VM

- Reference: https://cloud.google.com/tpu/docs/creating-deleting-tpus
- `gcloud compute ssh tewh-malaria --zone=us-central1-c`
- `export TPU_NAME=tewh-malaria`
- `exit`
- Confirm with `ctpu status` that the instances are RUNNING

### Remember to delete VM + TPU after training to save on costs!

- Reference: https://cloud.google.com/tpu/docs/creating-deleting-tpus
- `gcloud compute instances delete tewh-malaria --zone=us-central1-c`
- `gcloud compute tpus delete tewh-malaria --zone=us-central1-c`


## Jupyter Notebook Installation


### SSH into the VM

- `gcloud compute ssh tewh-malaria --zone=us-central1-c`


### Clone the Repository

- `cd /srv`
- `git clone [???] jupyter`


### Install Docker

- docker
- docker-compose


### Start the Container

- `cd /srv/jupyter`
- `docker-compose up`
