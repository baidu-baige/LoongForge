# ARG BASE_IMAGE="registry.baidubce.com/cce-ai-native/nvidia/pytorch:23.12-py3_aiak_training_llm_0418"
# 社区base版本：registry.baidubce.com/cce-ai-native/nvidia/pytorch:23.12-py3
# 增加依赖包版本：registry.baidubce.com/cce-ai-native/nvidia/pytorch:23.12-py3_aiak_training_llm_0417
# 增加补丁版本，设置 NVTE_FUSED_ATTN=0：registry.baidubce.com/cce-ai-native/nvidia/pytorch:23.12-py3_aiak_training_llm_0418
# ARG BASE_IMAGE="registry.baidubce.com/aiak-training-ns/nvidia/pytorch:24.10-py3"
# ARG BASE_IMAGE="registry.baidubce.com/aiak-training-ns/nvidia/pytorch:25.06-py3"

# 升级基础镜像，cuda 13.0、torch 2.8、deepspeed0.18.2、cudnn9.12.0.46-1
# 当前版本下 DeepEP 编译报错，暂不升级
# ARG BASE_IMAGE="registry.baidubce.com/aiak-training-ns/nvidia/pytorch:25.08-py3-deepspeed0.18.2"

# 升级基础镜像，cuda 12.9、torch 2.8、deepspeed0.17.6
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

RUN mkdir -p /workspace 
COPY ./AIAK-Training-Omni /workspace/AIAK-Training-Omni
RUN bash /workspace/AIAK-Training-Omni/docker/install.sh ${COMPILE_ENV} \
    ${BINARY_REPLACE} ${INSTALL_TransformerEngine_FLAG} ${MEGATRON_TYPE} ${INSTALL_LEROBOT}