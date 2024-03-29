
FROM nvcr.io/nvidia/cuda:12.3.1-base-ubi9

ENV CONFIG_FILE_NAME config.yaml
ENV RAY_SERVE_ENABLE_EXPERIMENTAL_STREAMING 1

# Install python 
RUN yum update -y && \
    yum install -y \
    python \
    python-devel && \
    rm -rf /var/cache/yum

# Install python libraries
COPY requirements.txt /tmp/requirements.txt
RUN pip3 --no-cache-dir install -r /tmp/requirements.txt

# Create folder for models weights
RUN mkdir -p /models

# Support arbitrary user ids
RUN chgrp -R 0 /models && \
    chmod -R g=u /models

#Copy project files
RUN mkdir -p /llm_api_server
COPY llm_api_server/ /llm_api_server

# Support arbitrary user ids
RUN chgrp -R 0 /llm_api_server && \
    chmod -R g=u /llm_api_server

WORKDIR /llm_api_server

EXPOSE 7000

CMD  ["sh", "-c", "ray start --head --port=6379 --object-manager-port=8076 --dashboard-host=0.0.0.0 && serve run $CONFIG_FILE_NAME"]
