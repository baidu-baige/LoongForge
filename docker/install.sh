#!/bin/bash

##########################################
#
# 1. 将项目copy 到容器，安装依赖
# 2. 增加参数判断是否需要替换二进制
#
##########################################
set -exuo pipefail

############################################################# 基础环境开始 ###########################################################

function install_pip_config() {
    cp pip.conf /etc/pip.conf
    cp pip.conf /usr/pip.conf
    mkdir -p /root/.pip && cp pip.conf /root/.pip/pip.conf
    mkdir -p /etc/xdg/pip && cp pip.conf /etc/xdg/pip/pip.conf
}

function install_base_env() {
    local COMPILE_ENV
    read COMPILE_ENV<<< $*

    cd ${CURRENT_DIR}
    rm -rf /opt/conda/pip.conf /root/.config/pip/pip.conf /root/.pip/pip.conf /etc/pip.conf /etc/xdg/pip/pip.conf /usr/pip.conf

    # install pkg
    install_requirements

    # pip config
    mkdir -p /tmp && chmod -R 777 /tmp
    install_pip_config

    rm -rf /etc/apt/sources.list && cp /workspace/AIAK-Training-Omni/docker/sources.list /etc/apt/sources.list

    # 设置时区
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install tzdata && ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
    apt-get install bc tree pwgen nodejs -y

    # 安装 sshd
    #echo "y" | apt-get install openssh-client=1:8.2p1-4ubuntu0.12
    #echo "y" | apt-get install openssh-sftp-server
    #echo "y" | apt-get install openssh-server

    # 创建 SSH 目录并设置权限
    #sed -i 's/[ #]\(.*StrictHostKeyChecking \).*/ \1no/g' /etc/ssh/ssh_config
    #echo "    UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config
    #sed -i 's/#\(StrictModes \).*/\1no/g' /etc/ssh/sshd_config

    #mkdir /var/run/sshd -p
}

function install_requirements() {
    # 移除镜像中的 contraints, 以便升级库
    source_file="/etc/pip/constraint.txt"
    if [ -f "$source_file" ]; then
        backup_file="${source_file%.*}_backup.${source_file##*.}"
        mv -v "$source_file" "$backup_file"
        > "$source_file"
    fi

    # 安装依赖包
    requirements_file="/workspace/AIAK-Training-Omni/requirements.txt"
    if [[ "$COMPILE_ENV" == "p800" ]];then
        requirements_file="/workspace/AIAK-Training-Omni/requirements_xpu.txt"

        echo "alias ll='ls -alF'" > /etc/profile.d/alias.sh
        echo "source /etc/profile.d/alias.sh" >> ~/.bashrc
    fi
    pip install wandb -i http://pip.baidu.com/pypi/simple --trusted-host pip.baidu.com
    pip install swanlab==0.6.1 -i http://pip.baidu.com/pypi/simple --trusted-host pip.baidu.com 
    pip install -r ${requirements_file} -i http://mirrors.baidubce.com/pypi/simple --trusted-host mirrors.baidubce.com
    
    # 安装 FA3，适配 TE
    update_flash_attn3_import_path_for_te
    # 安装 Deepep
    install_deepep
    # 安装 AIAK FP8量化算子，用于支持权重转换
    install_aiak_fp8_quant

    # 临时卸载 ngc 默认集成的 nvidia-modelopt，量化库，当前暂不需要；另外和 te 当前版本存在兼容性报警
    echo "y" | pip uninstall nvidia-modelopt
}

function update_flash_attn3_import_path_for_te() {
    # 兼容 te 对 fa 算子的调用方式
    python_path=`python -c "import site; print(site.getsitepackages()[0])"`
    flash_attn_3_path=$python_path/flash_attn_3/
    mkdir -p $flash_attn_3_path

    if [ -f "$python_path/flash_attn_interface.py" ]; then
        cp $python_path/flash_attn_interface.py $flash_attn_3_path
        echo "install fa3 success!"
    else
        echo "$python_path/flash_attn_interface.py not found, fa3 may not be installed!"
    fi
}

function install_deepep() {
    cd ${CURRENT_DIR}
    # from mk
    wget -q https://cce-ai-datasets.bj.bcebos.com/hac-aiacc/aiak-training-llm/ngc2504_torch27_cuda129_tev24/nvshmem.tar.gz
    tar -zxvf nvshmem.tar.gz
    cp -rf nvshmem /usr/local/nvshmem

    export NVSHMEM_DIR=/usr/local/nvshmem/
    export LD_LIBRARY_PATH=${NVSHMEM_DIR}/lib:$LD_LIBRARY_PATH
    export PATH=${NVSHMEM_DIR}/bin:$PATH

    rm -rf nvshmem
    rm -f nvshmem.tar.gz

    # 基于 https://github.com/deepseek-ai/DeepEP/  2025年4月20日版本
    wget -q https://cce-ai-datasets.bj.bcebos.com/hac-aiacc/aiak-training-llm/ngc2504_torch27_cuda129_tev24/DeepEP.tar.gz
    tar -zxvf DeepEP.tar.gz

    cd ${CURRENT_DIR}/DeepEP

    # deepep only support hopper now
    TORCH_CUDA_ARCH_LIST="9.0" python setup.py install

    cd ${CURRENT_DIR}

    rm -rf DeepEP
    rm -f DeepEP.tar.gz
}

function install_aiak_fp8_quant() {
    cd ${CURRENT_DIR}

    # 基于 https://console.cloud.baidu-int.com/devops/icode/repos/baidu/hac-aiacc/AIAK-FP8-Quantization/tree/f83ca42af556a0a7c6b659ecacfd1b1e4a8eab1c/
    wget -q https://cce-ai-datasets.bj.bcebos.com/hac-aiacc/aiak-training-llm/ngc2504_torch27_cuda129_tev24/AIAK-FP8-Quantization.tar.gz
    tar -zxvf AIAK-FP8-Quantization.tar.gz

    cd ${CURRENT_DIR}/AIAK-FP8-Quantization/

    python setup.py clean
    python setup.py build_ext --inplace --force
    pip install .

    cd ${CURRENT_DIR}

    rm -rf AIAK-FP8-Quantization/
    rm -f AIAK-FP8-Quantization.tar.gz
}

function clear_unused_file() {
    # 删除 /tmp/* ~/.bash_history
    rm -rf /tmp/* ~/.bash_history

    # 删除git相关
    rm -rf /workspace/AIAK-Training-Omni/.git

    # 删除不对外暴露的文件
    rm -rf /workspace/AIAK-Training-Omni/build.sh
    rm -rf /workspace/AIAK-Training-Omni/ci.yml
    rm -rf /workspace/AIAK-Training-Omni/docker
    rm -rf /workspace/AIAK-Training-Omni/ci
    rm -rf /workspace/TransformerEngine
    rm -rf /workspace/AIAK-ACCELERATOR

    rm -rf /workspace/AIAK-Megatron/.git
    rm -rf /workspace/AIAK-Megatron/.github
    rm -rf /workspace/AIAK-Megatron/ci.yml
    rm -rf /workspace/AIAK-Megatron/build.sh
    rm -rf /workspace/AIAK-Megatron/output
}

function download_megatron_binary_compile() {
    # 离线包 cpu_adam.so 导入
    rm -rf /workspace/AIAK-Megatron/megatron/core/optimizer/hybrid_adam/builder
    rm -rf /workspace/AIAK-Megatron/megatron/core/optimizer/hybrid_adam/csrc
    wget https://cce-ai-datasets.bj.bcebos.com/hac-aiacc/aiak-training-llm/binary_compile/cpu_adam.so -q -O /workspace/AIAK-Megatron/megatron/core/optimizer/hybrid_adam/cpu_adam.so
}

function encrypt_code() {
    cd  /workspace/AIAK-Megatron
    cp tools/cython_setup.py ./
    python cython_setup.py build
    rm -rf cython_setup.py tools/cython_setup.py tools/authority
}

function install_transformerEngine() {
    local transformerEngine_dir=/workspace/TransformerEngine
    cd $transformerEngine_dir

    # # 在线安装依赖：该代理网络掉线频繁，流水线会经常失败，所以暂时不使用
    # export http_proxy=http://gzbh-aip-paddlecloud140.gzbh:8128
    # export https_proxy=http://gzbh-aip-paddlecloud140.gzbh:8128
    # export GIT_SSL_NO_VERIFY=1
    # git submodule update --init --recursive
    # unset http_proxy https_proxy

    # 离线安装依赖，离线包需要手动线下拉取 github 最新的压缩成tar包上传到bos上；
    thirdparty_path="${transformerEngine_dir}/3rdparty"
    rm -rf ${thirdparty_path} && mkdir -p ${thirdparty_path}
    cd ${thirdparty_path}

    wget -q https://cce-ai-datasets.bj.bcebos.com/hac-aiacc/aiak-training-llm/ngc2504_torch27_cuda129_tev24/cudnn-frontend.tar.gz && tar -zxvf cudnn-frontend.tar.gz
    wget -q https://cce-ai-datasets.bj.bcebos.com/hac-aiacc/aiak-training-llm/ngc2504_torch27_cuda129_tev24/googletest.tar.gz && tar -zxvf googletest.tar.gz

    rm -rf cudnn-frontend.tar.gz googletest.tar.gz

    cd $transformerEngine_dir
    export NVTE_WITH_USERBUFFERS=1 && \
    export MPI_HOME=/usr/local/mpi && \
    export NVTE_FRAMEWORK=pytorch && \
    pip3 install .

    cd /workspace
    
    rm -rf $transformerEngine_dir
}

function install_AIAK_ACCELERATOR() {
    local AIAK_ACCELERATOR_tmp="/workspace/AIAK-ACCELERATOR"
    local AIAK_ACCELERATOR_dir="/usr/local/lib/AIAK-ACCELERATOR"
    rm -rf ${AIAK_ACCELERATOR_dir}
    mv ${AIAK_ACCELERATOR_tmp} ${AIAK_ACCELERATOR_dir}
    cd ${AIAK_ACCELERATOR_dir}
    pip install -r requirements.txt
    pip install -e .
}

function install_bccl_env() {
    local BCCL_DOWNLOAD_ADDR=$1
    cd /workspace
    eval "$BCCL_DOWNLOAD_ADDR"
    tar -zxvf output.tar.gz
    cd ./output/aiak_v2_megatron/aiak-v2-megatron
    # dpkg -r libnccl2 libnccl-dev  # 这个用来卸载 aiak 老的包
    dpkg -i bccl-*.amd64.deb
    # 安装nc
    apt-get install netcat-openbsd
    rm -rf /workspace/output*
}

function install_community_megatron() {
    # export http_proxy=http://gzbh-aip-paddlecloud140.gzbh:8128
    # export https_proxy=http://gzbh-aip-paddlecloud140.gzbh:8128
    export http_proxy=http://agent.baidu.com:8891
    export https_proxy=http://agent.baidu.com:8891
    export no_proxy=mirrors.baidubce.com
    export GIT_SSL_NO_VERIFY=1
    cd /workspace && git clone https://github.com/NVIDIA/Megatron-LM.git
    cd / && git clone https://github.com/meta-llama/llama3.git

    unset http_proxy https_proxy
    
    cd llama3 && pip install -e . && cd ..
    pip install lamini mistral-common
    mkdir -p /workspace/Megatron-LM/train_examples
    cp -r /workspace/AIAK-Training-Omni/ci/tests/scripts/megatron_core/examples/* /workspace/Megatron-LM/train_examples/
    cp -r /workspace/AIAK-Training-Omni/ci/tests/scripts/megatron_core/training.patch /workspace/Megatron-LM/training.patch
    find /workspace/Megatron-LM/train_examples -type f -name "*.sh" -exec chmod +x {} \;

    # 记录当前版本的信息
    cd /workspace/Megatron-LM

    # 获取当前项目的最新commit ID
    commit_id=$(git log -1 --pretty=format:"%H")

    # 获取该commit的提交日期和时间
    commit_date=$(git show -s --format="%ci" $commit_id | head -n 1)

    # 将commit ID和日期时间记录到文本文件中
    echo "#############################################################" >> commit_record.txt
    echo "Commit ID: $commit_id" >> commit_record.txt
    echo "Commit Date and Time: $commit_date" >> commit_record.txt
    echo "#############################################################" >> commit_record.txt

    # 增加补丁代码：打印 token 吞吐指标
    # 生成补丁文件方式
    # training.py：原始代码
    # training_new.py：修改后的代码
    # diff -u /workspace/Megatron-LM/megatron/training/training.py /workspace/Megatron-LM/training_new.py > /workspace/Megatron-LM/training.patch

    # 应用补丁
    patch -p0 /workspace/Megatron-LM/megatron/training/training.py < /workspace/Megatron-LM/training.patch
    
    # 查看补丁后的diff
    cd /workspace/Megatron-LM
    git diff /workspace/Megatron-LM/megatron/training/training.py

}

function install_aiak_training_omni() {
    # 将 examples 目录下所有脚本的权限设置为可执行
    find /workspace/AIAK-Training-Omni/examples -type f -name "*.sh" -exec chmod +x {} \;

    # 修复训练多个 datasets 报错场景
    cd /workspace/AIAK-Megatron/megatron/core/datasets
    make -C .
}

function install_aihclite_jupyter() {
    # 支持开发机自定义镜像增加启动脚本
    mkdir -p /root/.jupyter
    cp /workspace/AIAK-Training-Omni/docker/enterpoint.sh /root/.jupyter/enterpoint.sh
    chmod +x /root/.jupyter/enterpoint.sh
}

# 参数1：编译GPU环境, h800/a800/p800
# 参数2：是否替换二进制
# 参数3：是否对代码加密
# 参数4：是否安装bccl
# 参数5：是否指定 bccl 地址
# 参数6：是否安装 TransformerEngine
# 参数7：是否安装 AIAK_ACCELERATOR
# 参数8：安装 megatron 的类别: aiak/community
COMPILE_ENV=$1
BINARY_REPLACE=$2
ENCRYPT_FLAG=$3
INSTALL_BCCL_FLAG=$4
INSTALL_BCCL_ADDR=$5
INSTALL_TransformerEngine_FLAG=$6
INSTALL_AIAK_ACCELERATOR_FLAG=$7
MEGATRON_TYPE=$8

echo "Received COMPILE_ENV: ${COMPILE_ENV}"

VERSION=""
TOKEN=""
if [ -n "${9:-}" ]; then
    VERSION=$9
fi
if [ -n "${10:-}" ]; then
    TOKEN=${10}
fi

CURRENT_DIR=$(cd `dirname $0`; pwd)
BCCL_VERSION=${BCCL_VERSION-"1.2.7.2"}
IREPO_TOKEN=${BCCL_IREPO_TOKEN-"1bebc022-2d71-41a7-896c-53b32131285f"}
if [[ "$VERSION" != "" ]];then
    BCCL_VERSION=$VERSION
fi
if [[ "$TOKEN" != "" ]];then
    IREPO_TOKEN=$TOKEN
fi

BCCL_DOWNLOAD_ADDR="wget -q -O output.tar.gz --no-check-certificate --header "IREPO-TOKEN:${IREPO_TOKEN}" "https://irepo.baidu-int.com/rest/prod/v3/baidu/AIPod/BCCL/releases/${BCCL_VERSION}/files""
echo "$BCCL_DOWNLOAD_ADDR"

# 如果是 p800 镜像，激活当前 conda 环境
if [[ "$COMPILE_ENV" == "p800" ]];then
    . /root/miniconda/etc/profile.d/conda.sh
    conda activate python38_torch201_cuda
fi

if [[ "$INSTALL_BCCL_ADDR" != "" ]];then
    BCCL_DOWNLOAD_ADDR=$INSTALL_BCCL_ADDR
fi

if  [[ "$BINARY_REPLACE" == "true" ]];then
    download_megatron_binary_compile
elif  [[ "$BINARY_REPLACE" == "all" ]];then
    install_base_env $COMPILE_ENV
    download_megatron_binary_compile
elif  [[ "$BINARY_REPLACE" == "false" ]];then
    install_base_env $COMPILE_ENV
fi

if  [[ "$ENCRYPT_FLAG" == "true" ]];then
    encrypt_code
fi

if  [[ "$INSTALL_BCCL_FLAG" == "true" ]];then
    install_bccl_env "$BCCL_DOWNLOAD_ADDR"
fi

if  [[ "$INSTALL_TransformerEngine_FLAG" == "true" ]];then
    install_transformerEngine
fi

if  [[ "$INSTALL_AIAK_ACCELERATOR_FLAG" == "true" ]];then
    install_AIAK_ACCELERATOR
fi

if  [[ "$MEGATRON_TYPE" == "community" ]];then
    install_community_megatron
else
    install_aiak_training_omni
fi

install_aihclite_jupyter

# 清理无用文件
clear_unused_file



############################################################# 基础环境结束 ###########################################################
