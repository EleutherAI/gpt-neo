# INSTALL

## Install Docker compose
```
sudo curl -L https://github.com/docker/compose/releases/download/1.21.2/docker-compose-`uname -s`-`uname -m` -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
docker-compose --version
```

## Create Optimize TF Docker Image
docker build --rm -t omniboard -


## Launch omniboard (background)
docker-compose up -d 

## Run shell for experiment
./scripts/run_docker.sh

then check if there are TPU available:
```bash
pu ls
```
or create a new one (ask first)
```
export NAME=${1:-tpu-v3-256}

gcloud compute tpus create $NAME \
    --accelerator-type v3-256 \
    --preemptible \
    --zone europe-west4-a \
    --network default \
    --project $(gcloud config get-value project) \
    --version 1.15.2
```

