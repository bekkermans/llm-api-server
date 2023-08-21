#!/bin/bash

cp config.yaml.examples config.yaml

podman run --rm -d \
    -p 7000:7000 \
    -p 8265:8265 \
    -e CONFIG_FILE_NAME=config.yaml \
    -v $(pwd)/config.yaml:/llm_api_server/config.yaml \
    -v $(pwd)/models:/models \
    -e NVIDIA_VISIBLE_DEVICES=all \
    --security-opt=label=disable \
    --cgroup-manager=cgroupfs \
    --pids-limit=5000 \
    quay.io/rhn_support_sbekkerm/llm-api-server:latest
