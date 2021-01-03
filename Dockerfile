FROM 3.9-alpine
WORKDIR /jannet
ADD requirements.txt .
RUN python3 -m pip install -r requirements.txt
ENTRYPOINT ["/bin/bash"]
