ARG BASE_IMAGE="registry.baidubce.com/aiak-training-ns/nvidia/pytorch:25.06-py3-deepspeed0.17.6"
FROM ${BASE_IMAGE}

# 支持根据卡类型获取算子包
ARG COMPILE_ENV={{COMPILE_ENV}}
# 默认: false, 设置true 则替换编译算子二进制
ARG BINARY_REPLACE="false"

# 默认: false, 设置true 是否安装 TransformerEngine
ARG INSTALL_TransformerEngine_FLAG="false"

# 安装 megatron 的类别: aiak/community
ARG MEGATRON_TYPE="community"

# 是否安装 lerobot: true/false
ARG INSTALL_LEROBOT="false"

# 指定token
ARG TOKEN=""

# Xpytorch的源路径（可选，默认值在OmniTraining/xpytorch_info.txt）
ARG XPYTORCH_URL_ARG=""

RUN mkdir -p /workspace
# NOTE: 由于icode未修改名称
COPY ./AIAK-Training-Omni /workspace/OmniTraining
RUN bash /workspace/OmniTraining/docker/install.sh ${COMPILE_ENV} \
    ${BINARY_REPLACE} ${INSTALL_TransformerEngine_FLAG} ${MEGATRON_TYPE} ${INSTALL_LEROBOT} "${XPYTORCH_URL_ARG}"