



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

The offline packing flow is WDS-native: it reads uncompressed source `*.tar`
WebDataset shards directly, builds a small manifest/pack plan, and writes packed
WebDataset shards. It does not require unpacking the source WDS into millions of
loose JSON/media files.

Supported V1 scenario:

| Data Scenario                              | sample_type         | Sample Class Name |
|--------------------------------------------|---------------------|-------------------|
| Offline packed mixed image/video, multi-image QA | packed_multi_mix_qa | CrudeSample |



```sh
cd LoongForge/tools/data_preprocess/vlm/offline_packing
```

Configure `configs/config.yaml`

```yaml
data:
  input_format: "wds"
  wds_dir: "/path/to/source_wds"
  template_text_key: "texts"
  work_dir: "/path/to/packing_work"
  packed_wds_dir: "/path/to/packed_wds"

model:
  model_type: "kimi-k2.5"
  # Use auto_processor for VLM datasets. auto_tokenizer is enough for text-only datasets.
  processor_loader: "auto_tokenizer"
  # Kimi production packing uses the released HF chat template directly.
  # No handwritten Kimi fallback template is used.
  use_hf_chat_template: true
  chat_template_path: "/mnt/cfs_bj_mt/mokai/checkpoints/Kimi-k25/chat_template.jinja"
  chat_template_kwargs:
    thinking: false
  processor_kwargs:
    # Use a path whose basename has no "." when trust_remote_code is needed.
    pretrained_model_name_or_path: "/path/to/official/huggingface_checkpoint/Kimi-k25"
    min_pixels: 3136   # 4*28*28
    max_pixels: 4014080 # 5120*28*28(4014080,8192)
    trust_remote_code: True
    use_fast: False

# Media file preprocessing configuration. Available preprocessing functions are defined in wds_pack/media/preprocess.py
media_preprocess:
  image: custom_image_preprocess

sample:
  max_token_len: 65536
  sample_type: packed_multi_mix_qa

packing:
  algorithm: "best_fit_decreasing"
  validate_pack_plan: false

artifacts:
  debug_artifacts: false
  keep_intermediate: false

packed_wds:
  maxsize: 2000000000
  maxcount: 1000000000

process:
  workers: 1
  max_shards: 0

log:
  level: "INFO"
```

Run

```sh
bash scripts/pack_wds.sh configs/config.yaml
```

Production outputs with `debug_artifacts: false` and `keep_intermediate: false`:

| File | Description |
|------|-------------|
| `sample_manifest.sqlite` | Authoritative source sample metadata and tar byte locators |
| `bins/bins_plan_{text,image,video}.jsonl` | Compact per-media packing plan consumed by `wds_pack.cli.build_plan` |
| `pack_plan.jsonl` | Final packed sample plan, including `sample_ids`, `sample_token_lens`, and `total_token_len` |
| `skipped_overlong.jsonl` | Samples skipped because `token_len > max_token_len` |

When `debug_artifacts: true`, the pipeline also keeps
`sample_manifest.jsonl`, `sample_len_report.txt`, per-media reports under
`token_len/`, `skipped_samples.jsonl`, and `unpacked_samples.jsonl`. When
`keep_intermediate: true`, it also keeps `manifest_parts/` and
`bins/bins_boxs_{text,image,video}.pkl`.

Packing is homogeneous by media type. Text-only samples are packed only with
text-only samples; image+text samples are packed only with image+text samples;
video+text samples are packed only with video+text samples.

Source layout:

```text
offline_packing/
  configs/                 # Example and production YAML configs
  scripts/                 # Shell entrypoints
  wds_pack/
    cli/                   # Step entrypoints
    core/                  # Config, paths, artifact flags, typed records
    manifest/              # WDS scan and SQLite manifest storage
    algorithms/            # Packing algorithms
    media/                 # Media preprocessing helpers
    io/                    # WebDataset writing
    legacy/                # Old flat-file packing tools
```

Python code lives under `wds_pack/`. Import from `wds_pack.*` or run the module
entrypoints:

```sh
python -m wds_pack.cli.scan_manifest --config configs/config.yaml
python -m wds_pack.cli.pack_bins --config configs/config.yaml
python -m wds_pack.cli.build_plan --config configs/config.yaml
python -m wds_pack.cli.write_wds --config configs/config.yaml
```

Implementation notes:

- The WDS-native path uses typed manifest records (`ManifestSample`,
  `ManifestMember`, `PackItem`) and SQLite helper APIs in
  `wds_pack.manifest.sqlite`.
  The packing stage does not parse `sample_len_report_*.txt`.
- `packing.algorithm: best_fit_decreasing` is the production path for long
  contexts such as 64k. It keeps all valid samples and allows non-exact bins;
  the short remainder is represented by `_meta.total_token_len` and can be
  padded by training. `packing.algorithm: hashbucket` is still available for
  the old exact-fill search behavior.
- `packing.validate_pack_plan: false` skips an extra manifest scan in
  `wds_pack.cli.build_plan` for BFD. The compact plan is generated from the
  manifest in Step 2, and Step 4 still checks sample ids against the manifest
  before writing WDS.
- `artifacts.debug_artifacts: false` disables human-readable diagnostic files.
  `artifacts.keep_intermediate: false` removes handoff files that are not needed
  after the next stage.
- Packed WDS JSON contains `_meta.pack_id`, `_meta.sample_ids`,
  `_meta.token_lens`, and `_meta.total_token_len` so a written packed sample can
  be audited without joining back to `pack_plan.jsonl`.

## Acknowledgements

The WDS-native offline packing workflow in LoongForge is based on the multimodal
offline packing framework originally developed for LLaVA-OneVision-1.5 and later
migrated and upgraded for LLaVA-OneVision-2.

LoongForge previously collaborated with the LLaVA-OneVision work. Some historical
repository or package names may still use the older `aiak-*` naming, while the
current LoongForge repository has migrated and adapted part of the
LLaVA-OneVision offline packing capabilities.

Upstream references:

- LLaVA-OneVision-1.5 offline packing:
  https://github.com/fdcp/LLaVA-OneVision-1.5/tree/main/tools/data_preprocess/offline_packing
- LLaVA-OneVision-1.5 offline packing examples:
  https://github.com/fdcp/LLaVA-OneVision-1.5/tree/main/examples_offline_packing
- LLaVA-OneVision-2 offline packing:
  https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-2/tree/main/offline_packing
- LLaVA-OneVision-2 sample packing scripts:
  https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-2/tree/main/examples/llava_onevision1_5/sample_packing

LoongForge refactors this workflow for native WebDataset tar-shard input,
manifest/SQLite-based sample indexing, media-type-specific packing, pack-plan
generation, tar byte-offset based WebDataset writing, and runtime handling for
packed text/image/video samples.
