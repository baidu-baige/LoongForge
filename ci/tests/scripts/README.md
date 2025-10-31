Megatron Training Example
=========================

此脚本提供在Kubernetes 集群/容器内运行Megatron 框架训练测试以及自动化验证工具

## 一、训练前准备阶段

### 基本信息

运行Megatron 大模型训练需要使用模型 Tokenizer/Datasets/Checkpoint 文件。目前流水线上运行训练的基本信息如下：

* 集群信息：`hwl-cce【cce-e0isdmib】`
* PFS 根目录： `/mnt/pfs/leoli`

### 1.1 已支持模型

如果需要运行的模型已在 `models_parameters` 目录下，证明已完成依赖数据下载，跳过准备阶段，直接进入到 `启动训练`即可

```
# 当前已支持的模型：
chatglm-6b
galactica-30b
galactica-6.7b
glm-10b-chinese
llama-13b
llama-7b

... 待增加
```

### 1.2 增加新模型

**增加新模型并在K8S 集群运行的流程如下：**

1. 准备新模型的Tokenizer/Datasets/Checkpoint 文件，并存储放置到 BOS 上，以llama-7b为例

* Tokenizer：bos:/cce-ai-datasets/cce-ai-datasets.bj.bcebos.com/megatron_llama/llama_tokenizer
* Datasets：bos:/cce-ai-datasets/cce-ai-datasets.bj.bcebos.com/megatron_llama/pile_llama_test
* Checkpoint：bos:/cce-ai-datasets/cce-ai-datasets.bj.bcebos.com/megatron_llama/megatron_llama_7b_checkpoint_tp1_pp1_dp8_zero1

2. 登录PFS所挂载的BCC机器，首次需要下载bcecmd 工具
3. 使用bcecmd 工具下载 Tokenizer/Datasets/Checkpoint 文件到 PFS `/mnt/pfs/leoli`目录下，具体可参考`download_datasets.sh` 文件内容。

```
⏰⏰⏰
- 新模型依赖文件最好同步到【download_datasets.sh】里，以方便其他同学快速下载使用哦～
```

4. 在 models_parameters 文件夹下增加新模型的训练参数。特别注意：TOTAL_K8S_NODES 参数需要与训练场景绑定，含义: 使用K8S 集群训练使用的 Node 总数
5. 提交代码，触发流水线构建镜像-执行测试即可

## 二、启动训练

### 2.1 流水线运行（K8S 集群）

作用：K8S 集群上创建 PytorchJob 训练任务

```
bash ipipe_start.sh

# 参数说明：
# case_type: 测试类型, test_function | test_perf 分别为功能/性能测试
# kubectl_path: kubectl 所在路径
# kubectl_view_allocations_path: kubectl_view_allocations 所在路径
# kubeconfig_path: kubeconfig 所在路径# gpu_resource: 资源描述符，比如：baidu.com/a100_80g_cgpu
# TRAIN_DATA_DIR: 模型依赖文件存在 pfs 上的根目录, 比如: /mnt/pfs/leoli
# IMAGE: 创建 PytorchJob 训练任务使用的镜像地址, 比如: registry.baidubce.com/hac_test/aiak-megatron:dev_20231010_144223
# TIMEOUT: 容器所有任务等待最大时间, 比如: 1800 秒
```

#### 2.1.1 性能数据查看

功能测试/性能测试阶段任务执行成功后，手动下载AIAK-Megatron 模型性能指标

![image](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=689bdbdc2b5242d8b079d906f4ed688c&docGuid=mDElYvVvogBoiq&sign=eyJhbGciOiJkaXIiLCJlbmMiOiJBMjU2R0NNIiwiYXBwSWQiOjEsInVpZCI6ImpkZlp6NWJnSU4iLCJkb2NJZCI6Im1ERWxZdlZ2b2dCb2lxIn0..7NbkOYYAstyZqb5a.34Kmt_-MTZUvECIggLzFBkkXElbfhdu8IMB4lG74kT1iOo2h_rE4_2DXI5msLtfCv31QbD6X7bxXTWumD6OAWLaivMU-8ly9Zmy1jR3y-O6U-smfdmUNgSmrXTyfuWtsDflA6HfgU4jCuRrN-xqFXSPbDYKFuWwEkfWOVdRzcY-qu7K4Cg_U8S3nR9uYaXeoHMmB35_I48I0zpd-TDGmdBEeyA.jA6i76LDBtTadSY9sDCtEQ&x-bce-process=image/resize,m_lfit,w_960/ignore-error,i_1)

![image](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=7886c69a4b1f415ba34b5c0abf1eee29&docGuid=PGdfhJcYjkpGPM&sign=eyJhbGciOiJkaXIiLCJlbmMiOiJBMjU2R0NNIiwiYXBwSWQiOjEsInVpZCI6ImpkZlp6NWJnSU4iLCJkb2NJZCI6Im1ERWxZdlZ2b2dCb2lxIn0..7NbkOYYAstyZqb5a.34Kmt_-MTZUvECIggLzFBkkXElbfhdu8IMB4lG74kT1iOo2h_rE4_2DXI5msLtfCv31QbD6X7bxXTWumD6OAWLaivMU-8ly9Zmy1jR3y-O6U-smfdmUNgSmrXTyfuWtsDflA6HfgU4jCuRrN-xqFXSPbDYKFuWwEkfWOVdRzcY-qu7K4Cg_U8S3nR9uYaXeoHMmB35_I48I0zpd-TDGmdBEeyA.jA6i76LDBtTadSY9sDCtEQ&x-bce-process=image/resize,m_lfit,w_960/ignore-error,i_1)

#### 2.1.2 评审/镜像版本/性能指标等通知

申请加入【AIAK-training 工程效能群】- 【群号：7466354】

![image](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=6d373531ce934b2c89cbb6af468c4e64&docGuid=PGdfhJcYjkpGPM&sign=eyJhbGciOiJkaXIiLCJlbmMiOiJBMjU2R0NNIiwiYXBwSWQiOjEsInVpZCI6ImpkZlp6NWJnSU4iLCJkb2NJZCI6Im1ERWxZdlZ2b2dCb2lxIn0..E-0GFNmcME5RnfMW.3CuGealFU0lz3vuVE2Y1B8KfLo3y67UZuTWa6NWfF7hvsgcOoKY8n6dZIzJyigKIc9pHSEjRowvana-Oz4Ra1GLeyUJ3v6Z-07_g4rOh3S44R5irXQxLF58el5bolUBdAZUSS4PpaZc7HnRx8VQH7HB8FXMbQrvIG3Z9a5E5x7WcJDTcSjWSPgPfAaWicRd2y72ts3Zo8HENPr8gDmewtMXDRg.UE59kEFxjU5LDzQio7kBdg)

### 2.2 容器内运行

作用：已有的容器或手动进入K8S Pod 后调试任务

```
bash local_start.sh

# 参数说明：
# case_type: 测试类型, test_function | test_perf 分别为功能/性能测试
# model_name: 模型名称, llama-7b
```

## 三、其他问题

### 0、流水线是如何提交训练任务到 K8S 集群上？

1. 找一台有存在 kubectl 的机器
2. 机器上放置好 kubeconfig 文件
3. 测试脚本替换训练参数到`yaml_template`文件夹下 pytorchjob yaml文件
4. 通过`kubectl apply -f {YAML文件名}`即可创建任务自动开始训练

```
# 流水线上默认的使用机器、kubectl、kubeconfig信息如下：
# kubectl 所在的机器：yq02-inf-hic-k8s-a100-ab2-0025.yq02
# kubectl 所在的目录：/usr/local/bin
# kubeconfig 所在的位置：/ssd4/leoli/kubeconfigs/hwl-cce.config
# kubectl_view_allocations_path 所在的目录：/usr/local/bin
```

![image](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=2849cf2033d44c859b96e61ab3a4a26f&docGuid=mDElYvVvogBoiq&sign=eyJhbGciOiJkaXIiLCJlbmMiOiJBMjU2R0NNIiwiYXBwSWQiOjEsInVpZCI6ImpkZlp6NWJnSU4iLCJkb2NJZCI6Im1ERWxZdlZ2b2dCb2lxIn0..G_zjV7JRcUbAurer.x71tXlM63EfRMTdCLl0FZuL2HjVu2W4F8JgqwKaaVBCa2dytlzM0MDRI8RJmZwDLUEIFt7km51EThyISDevWSoofOS0hnVlhpcJZzcEb-GhO9tglncvHhBc8Dxe9LsRdLDZ9BV6q63Ub3Er46R0NmvicdO_tDbyN5uBE_9R6yqdi7K4ghv4Wvsx2lnyHiCLosokx00BOFKn-lGxLeDy59j7P-A.P-IQF4G9k-JFTrZNbOrzOg)

### 1、流水线运行指定的单个模型如何操作？

1. [打开 Megatron 流水线](https://console.cloud.baidu-int.com/devops/ipipe/workspaces/398311/pipelines/1212289/builds/list?branchName=main)
2. 功能/性能测试阶段点击【重新执行】
3. 单选项【specific_model_name】选择需要的模型，如若没有点击右上角【流水线配置】增加新模型名称即可

![指定模型运行](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=55cd508edbad411e9c66b96e03e7c3f9&docGuid=mDElYvVvogBoiq&sign=eyJhbGciOiJkaXIiLCJlbmMiOiJBMjU2R0NNIiwiYXBwSWQiOjEsInVpZCI6ImpkZlp6NWJnSU4iLCJkb2NJZCI6Im1ERWxZdlZ2b2dCb2lxIn0..G_zjV7JRcUbAurer.x71tXlM63EfRMTdCLl0FZuL2HjVu2W4F8JgqwKaaVBCa2dytlzM0MDRI8RJmZwDLUEIFt7km51EThyISDevWSoofOS0hnVlhpcJZzcEb-GhO9tglncvHhBc8Dxe9LsRdLDZ9BV6q63Ub3Er46R0NmvicdO_tDbyN5uBE_9R6yqdi7K4ghv4Wvsx2lnyHiCLosokx00BOFKn-lGxLeDy59j7P-A.P-IQF4G9k-JFTrZNbOrzOg&x-bce-process=image/resize,m_lfit,w_960/ignore-error,i_1)

### 2、流水线启动脚本 ipipe_start.sh 中 case_type 的参数含义？

case_type 代表测试类型，分别功能和性能测试：test_function | test_perf

- test_function：功能测试，只运行模型 micro-batch-size=1、global-batch-size=8 场景
- test_perf：性能测试，micro-batch-size和global-batch-size随着models_parameters具体模型参数而使用，不做修改

### 3、kubectl、kubectl_view_allocations 二进制从哪里下载？

```
kubectl: https://kubernetes.io/zh-cn/releases/download/
kubectl_view_allocations: https://github.com/davidB/kubectl-view-allocations/releases
```

### 4、算子包如何上传到BOS？

```
# 将编译好的算子包压缩成tar 包
cp /workspace/Megatron-LM/megatron/fused_kernels/build/*.so ./
cp /workspace/Megatron-LM/megatron/data/helpers.cpython-*.so ./
tar -czf megatron_compile.tar.gz *.so

# A800 算子上传到bos
./bcecmd bos cp megatron_compile.tar.gz bos:/cce-ai-datasets/hac_test/megatron/binary_compile/a800/megatron_compile.tar.gz

# H800 算子上传到bos
./bcecmd bos cp megatron_compile.tar.gz bos:/cce-ai-datasets/hac_test/megatron/binary_compile/h800/megatron_compile.tar.gz
```

