# TEWH-Malaria-Google-Cloud-Setup

## Google Cloud Configuration

### Initial Steps

- Make a [Google Cloud account](https://cloud.google.com) and sign up for the free $300 credit
- Create a new project
- Navigate to Compute Engine -> [VM instances](https://console.cloud.google.com/compute/instances) in the left sidebar


### Start Preemptible VM Instance

- Reference: https://cloud.google.com/preemptible-vms
- `gcloud compute instances create tewh-malaria --machine-type=n1-standard-2 --accelerator type=nvidia-tesla-v100,count=1 --image-project=ml-images --image-family=tf-1-15 --zone=us-central1-a --maintenance-policy=TERMINATE`


### Remember to delete VM + TPU after training to save on costs!

- Reference: https://cloud.google.com/tpu/docs/creating-deleting-tpus
- `gcloud compute instances delete tewh-malaria --zone=us-central1-a`


## Jupyter Notebook Installation


### SSH into the VM

- `gcloud compute ssh tewh-malaria --zone=us-central1-a`


### Clone the Repository

- `cd /srv`
- `git clone [???] jupyter`


### Install Docker

- docker
- docker-compose


### Start the Container

- `cd /srv/jupyter`
- `docker-compose up`
