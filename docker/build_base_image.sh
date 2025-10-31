#!/bin/bash
# 用于构建基础镜像
# 目前包含安装 apex

set -exuo pipefail

# 启动容器
container_name=aiak_training_omni_base_image
nvidia-docker run -itd --name=${container_name} --rm registry.baidubce.com/cce-ai-native/nvidia/pytorch:23.12-py3 bash
docker exec -it ${container_name} bash

# 容器中安装apex
export http_proxy=http://agent.baidu.com:8118
export https_proxy=http://agent.baidu.com:8118
export no_proxy=mirrors.baidubce.com 

git clone https://github.com/NVIDIA/apex
cd apex
# if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key... 
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

unset http_proxy https_proxy no_proxy
rm -rf ~/.bash_history
rm -rf /root/.bash_history
rm -rf /tmp

# 退出容器
exit

# 生成新的镜像
container_id=$(docker ps -a -q --filter name=${container_name})
aiak_training_omni_base_image="registry.baidubce.com/cce-ai-native/nvidia/pytorch:23.12-py3_aiak_training_omni_0417"
docker commit -m "add apex" ${container_id} ${aiak_training_omni_base_image}
docker push ${aiak_training_omni_base_image}