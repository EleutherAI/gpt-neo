FROM ubuntu:latest
COPY requirements.txt install_packages.sh ./
RUN chmod +x install_packages.sh && ./install_packages.sh && rm install_packages.sh
RUN mkdir buffer datasets
COPY video2tfrecord.py ./
