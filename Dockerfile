FROM continuumio/miniconda3

WORKDIR /neogpt
ADD requirements.txt .
RUN conda install gcc_linux-64 gxx_linux-64 -y
RUN pip install -r requirements.txt

CMD /bin/bash