



## 1 JSONL 格式 转 webdataset 格式

支持数据集形式

| 数据场景          | sample_type  | sample 类名        |
|---------------|--------------|------------------|
| 单张图片VQA       | vqa          | VQASample        |
| 多个视频VQA       | multi_vid_qa | CrudeSample      |
| 图片视频混合、多张图片QA | multi_mix_qa | CrudeSample      |
| caption数据     | captioning     | CaptioningSample |


```sh
cd LoongForge/tools/data_preprocess

python convert_to_webdataset.py \
    --json_file /mnt/cluster/data/mmdu-45k.jsonl \
    --image_dir /mnt/cluster/data/images/ \
    #--video_dir /mnt/cluster/data/videos/
    --media_type image \
    --output_dir  /mnt/cluster/data/wds/ \
    --maxcount 10000 \
    --maxsize 100000 \
    --message_key conversations \
    --sample_type multi_mix_qa 
```

| 参数                   | 类型  | 默认             | 描述              |
|----------------------|-----|----------------|-----------------|
| `--output_dir`       | str | -              | 输出路径            |
| `--json_file`        | str | -              | json路径          |
| `--image_dir`        | str | None           | image文件路径       |
| `--video_dir`        | str | None           | video文件路径       |
| `--media`            | str | `image`        | image/video/mix |
| `--columns_messages` | str | `messages`     | json里面的消息key    |
| `--maxcount`         | int | 10000          | 每个shard最大的数量    |
| `--maxsize`          | int | 3000000000     | 每个shard最大的大小    |
| `--max_workers`      | int | CPU cores // 2 | 并行              |



## 2 离线packing

基于wds数据集，转换为packing后的 wds数据集，支持场景：

| 数据场景                  | sample_type         | sample 类名   |
|-----------------------|---------------------|-------------|
| 离线packed的caption      | packed_captioning   | CrudeSample |
| 离线packed的单图QA         | packed_vqa          | CrudeSample |
| 离线packed图片视频混合、多张图片QA | packed_multi_mix_qa | CrudeSample |



```sh
cd LoongForge/tools/data_preprocess/omni_packing

```

配置 `omni_packing/config.yaml`

```yaml
# 数据路径配置
data:
  # webdataset数据样本目录
  wds_dir: "/mnt/cluster/yxc/test_4/"
  # 文本字段名
  template_text_key: "messages"
  packed_json_dir: "/mnt/cluster/yxc/test_5/"
model:
  model_type: "qwenvl"
  processor_kwargs:
    pretrained_model_name_or_path: "/mnt/cluster/huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct"
    min_pixels: 3136   # 4*28*28
    max_pixels: 4014080 # 5120*28*28(4014080,8192)
    trust_remote_code: True
    use_fast: False

# 媒体文件预处理配置，可选的的预处理函数在media_preprocess_utils.py中定义
media_preprocess:
  image: custom_image_preprocess

sample:
  # 训练数据的最大长度
  max_token_len: 8192
  # 决定解析方式
  sample_type: packed_multi_mix_qa

# 并行处理参数
process:
  # 每个进程处理的样本块大小
  chunk_size: 5000
  # 归并参数（排序)的batch_size
  merge_batch_size: 20
  # 超时设置（根据数据量定，1M数据按 45分钟(2700s)估算）
  time_out: 2000

# 日志与临时文件
log:
  # 日志级别（DEBUG, INFO, WARNING, ERROR, CRITICAL）
  level: "INFO"
  # 日志文件路径
```

执行

```sh
sh pack_wds.sh
```