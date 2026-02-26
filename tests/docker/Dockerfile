ARG BASE_IMAGE="registry.baidubce.com/hac-aiacc/OmniTraining:ubuntu22.04-cu12.3-torch2.2.0-py310-bccl1.2.4.1_dev_20240424_201333_encrypt"
FROM ${BASE_IMAGE}

RUN rm -rf /workspace/OmniTraining/tests
RUN mkdir -p /workspace/OmniTraining/tests
COPY . /workspace/OmniTraining/tests