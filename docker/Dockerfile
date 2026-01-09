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

# 默认: false, 设置true 是否需要针对代码进行加密
ARG ENCRYPT_FLAG="false"

# 默认: false, 设置true 是否安装bccl
ARG INSTALL_BCCL_FLAG="false"

# bccl 下载地址
ARG INSTALL_BCCL_ADDR=""

# 默认: false, 设置true 是否安装 TransformerEngine
ARG INSTALL_TransformerEngine_FLAG="false"

# 默认: false, 设置true 是否安装 AIAK-ACCELERATOR
ARG INSTALL_AIAK_ACCELERATOR_FLAG="false"

# 安装 megatron 的类别: aiak/community
ARG MEGATRON_TYPE="aiak"

# 指定bccl版本号
ARG VERSION=""

# 指定token
ARG TOKEN=""

# 标志是否是BCCL 版本
ENV USE_BCCL=${INSTALL_BCCL_FLAG}

RUN if [ "${COMPILE_ENV}" = "p800" ] ; then rm -rf /workspace/Megatron-LM /workspace/aiak-tool ; else rm -rf /workspace/ ; fi
RUN mkdir -p /workspace 
COPY ./AIAK-Training-Omni /workspace/AIAK-Training-Omni
RUN bash /workspace/AIAK-Training-Omni/docker/install.sh ${COMPILE_ENV} \
    ${BINARY_REPLACE} ${ENCRYPT_FLAG} ${INSTALL_BCCL_FLAG} "${INSTALL_BCCL_ADDR}" \
    ${INSTALL_TransformerEngine_FLAG} ${INSTALL_AIAK_ACCELERATOR_FLAG} ${MEGATRON_TYPE} \
    "${VERSION}" "${TOKEN}"

# 临时补丁，解决报错: AssertionError: add_position_embedding value from checkpoint (True) is not equal to the input argument value (False)
# ENV NVTE_FUSED_ATTN=0

# 设置 bccl 安装路径
ENV BCCL_PATH=/usr/local/bccl/lib
