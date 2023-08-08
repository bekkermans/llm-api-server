#!/bin/bash

cp config.yaml.examples config.yaml

podman run --rm \
    -d \
    -v $(pwd)/config.yaml:/llm_api_server/config.yaml \
    -e CONFIG_FILE_NAME=config.yaml \
    --security-opt=label=disable \
    --cgroup-manager=cgroupfs \
    --pids-limit=5000 \
    localhost/llm_api_server:latest