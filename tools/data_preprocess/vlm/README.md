# VLM 数据集

## 1 数据集格式和处理
考虑多模态数据集的多样性，本次版本将采用 Energon 加载器来提升数据处理性能，它要求数据集以标准的 WebDataset 格式存储。WebDataset 是以原生文件格式 (jpg、mp4等) 存储数据，这使得各种原生的多模态数据集只需简单地压缩就能转成 WebDataset 格式，进而被 Energon 读取。
相关的参考文档：
* Energon： https://nvidia.github.io/Megatron-Energon/
* WebDataset：https://huggingface.co/docs/hub/datasets-webdataset


重点支持了  VQA 和 Captioning 两种多模态数据格式，下文分别介绍具体的数据处理流程：
* VQA格式的数据集，以Qwen2-VL 提供的VQA样本为例。
* Captioning格式的数据集，以 minigpt4_3500 数据集为例。


### 1.1 VQA 数据格式


样本来自 https://huggingface.co/datasets/THUDM/CogVLM-SFT-311K

```sh
python /workspace/AIAK-Training-LLM/tools/data_preprocess/convert_to_webdataset.py \
    --output_dir /tmp/mllm/wds \
    --json_file /tmp/mllm/mllm_demo.json \
    --image_dir /tmp/mllm/ \
    --maxcount 10000
```

功能说明：
* convert_to_webdataset.py会将 json_file 中每个样本抽出来存储成独立的json文件，并和 image_dir 中相对应的图片一起被压缩到 $output_dir，每个tar包最多包含有 maxcount 个样本；
* 后续启动训练时，将通过--data-path参数指定如上 WebDataset 路径  /tmp/mllm/wds，用于训练数据读取



执行示例：
* 当执行上面脚本后，--output_dir目录下新增处理后的数据压缩包：


pretrain-0.tar 中的文件如下，其中具有相同前缀的数据文件都属于同一个样本，比如 1.jpg 和 1.json 属于样本1。


### 1.2 Captioning 数据格式


准备原始数据集
样本来自 https://huggingface.co/datasets/THUDM/CogVLM-SFT-311K


```sh
cd /tmp/minigpt4
mkdir wds
tar -cf wds/pretrain.tar samples/
```


由于原始数据集格式，已经和convert_to_webdataset.py工具所生成 tar 内的格式一致，因此仅需要执行压缩命令即可，而不需要额外调用转换工具单独处理；


WebDataset 格式转成 Megatron-Energon 格式
```sh
cd wds
energon prepare ./
为所显示的选项选择以下值：
> Please enter a desired train/val/test split: 10,0,0
> Do you want to create a dataset.yaml interactively? [Y/n]: Y
> Please enter a number to choose a class: 0
> Please enter a webdataset field name for 'image' : jpg
> Please enter a webdataset field name for 'caption': json[captions][0][content]
Done
```


## 2 离线Packing数据处理

详见
[Offline Data Packing Guide](tools/data_preprocess/vlm/offline_packing/README.md)