FROM 3.7-alpine
WORKDIR /jannet
ADD requirements.txt .
RUN python3 -m pip install -r requirements.txt
ENTRYPOINT ["/bin/bash"]
