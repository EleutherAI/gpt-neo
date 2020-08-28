FROM continuumio/miniconda3

WORKDIR /neogpt

# Make RUN commands use `bash --login`:
SHELL ["/bin/bash", "--login", "-c"]
RUN conda install gcc_linux-64 gxx_linux-64 -y

ADD requirements.txt .
RUN pip install -r requirements.txt fasttext

CMD /bin/bash