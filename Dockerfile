FROM continuumio/miniconda3

WORKDIR /neogpt
ADD requirements.txt .
RUN pip install -r requirements.txt

CMD /bin/bash