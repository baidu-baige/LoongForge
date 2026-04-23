



## 1 Converting JSONL Format to WebDataset Format

Supported dataset types

| Data Scenario                              | sample_type  | Sample Class Name |
|--------------------------------------------|--------------|-------------------|
| Single image VQA                           | vqa          | VQASample         |
| Multiple video VQA                         | multi_vid_qa | CrudeSample       |
| Mixed image/video, multi-image QA          | multi_mix_qa | CrudeSample       |
| Caption data                               | captioning   | CaptioningSample  |


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

| Parameter            | Type | Default        | Description                   |
|----------------------|------|----------------|-------------------------------|
| `--output_dir`       | str  | -              | Output path                   |
| `--json_file`        | str  | -              | JSON file path                |
| `--image_dir`        | str  | None           | Image file path               |
| `--video_dir`        | str  | None           | Video file path               |
| `--media`            | str  | `image`        | image/video/mix               |
| `--columns_messages` | str  | `messages`     | Message key in the JSON file  |
| `--maxcount`         | int  | 10000          | Maximum number of samples per shard |
| `--maxsize`          | int  | 3000000000     | Maximum size per shard        |
| `--max_workers`      | int  | CPU cores // 2 | Parallelism                   |



## 2 Offline Packing

Based on WebDataset datasets, converts to packed WebDataset datasets. Supported scenarios:

| Data Scenario                              | sample_type         | Sample Class Name |
|--------------------------------------------|---------------------|-------------------|
| Offline packed caption                     | packed_captioning   | CrudeSample       |
| Offline packed single image QA             | packed_vqa          | CrudeSample       |
| Offline packed mixed image/video, multi-image QA | packed_multi_mix_qa | CrudeSample |



```sh
cd LoongForge/tools/data_preprocess/omni_packing

```

Configure `omni_packing/config.yaml`

```yaml
# Data path configuration
data:
  # WebDataset sample directory
  wds_dir: "/mnt/cluster/yxc/test_4/"
  # Text field name
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

# Media file preprocessing configuration. Available preprocessing functions are defined in media_preprocess_utils.py
media_preprocess:
  image: custom_image_preprocess

sample:
  # Maximum token length for training data
  max_token_len: 8192
  # Determines the parsing method
  sample_type: packed_multi_mix_qa

# Parallel processing parameters
process:
  # Chunk size of samples processed per worker
  chunk_size: 5000
  # Batch size for merge (sorting) parameter
  merge_batch_size: 20
  # Timeout setting (set based on data volume; estimate ~45 minutes (2700s) per 1M samples)
  time_out: 2000

# Logging and temporary files
log:
  # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
  level: "INFO"
  # Log file path
```

Run

```sh
sh pack_wds.sh
```