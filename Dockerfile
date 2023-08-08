
FROM quay.io/redhat_emp1/sbekkerm-ubi9

ENV CONFIG_FILE_NAME config.yaml
ENV RAY_SERVE_ENABLE_EXPERIMENTAL_STREAMING 1

# Install python 
RUN yum update -y && \
    yum install -y \
    python3 \
    python3-devel && \
    rm -rf /var/cache/yum

# Install python libraries
COPY requirements.txt /tmp/requirements.txt
RUN pip --no-cache-dir install -r /tmp/requirements.txt

# Copy models weight to the models folder
RUN mkdir -p /models
COPY models/ /models/

#Copy project files
RUN mkdir -p /llm_api_server
COPY llm_api_server/ /llm_api_server

#Copy config file
COPY ${CONFIG_FILE_NAME} /llm_api_server/${CONFIG_FILE_NAME}
WORKDIR /llm_api_server

EXPOSE 7000

CMD ["sh", "-c", "serve run $CONFIG_FILE_NAME"]