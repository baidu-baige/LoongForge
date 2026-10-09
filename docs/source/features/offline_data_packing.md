# Offline Packing  
This module provides an “offline sequence-packing” pipeline: it reads source **WebDataset tar shards** directly, groups and re-orders the samples according to `max_token_len`, and finally produces a **packed WebDataset** (`pretrain-*.tar` plus Energon meta files).
By concatenating variable-length sequences up to the target length we reduce padding and increase training throughput.

Entry script:  
`tools/vlm_data_preprocess/offline_packing/scripts/pack_wds.sh` (4 steps, see below).

## 1. Supported packing scenarios (`sample.sample_type`)

The WDS-native V1 path (`wds_pack.cli.scan_manifest`) accepts exactly two `sample.sample_type` values; any other value is rejected at scan time.

|Scenario|`sample_type`|Description|
|---|---|---|
|Offline packed image/video/text mixed QA|`packed_multi_mix_qa`|Input WDS JSON must declare `media`/`media_type`; packs are homogeneous by media type. Uses the handwritten `TEMPLATES[sample_type][model_type]` by default; if `model.use_hf_chat_template: true` or `model.chat_template_path` is set it renders with the HF chat template instead (the handwritten template is only the fallback when HF rendering is disabled).|
|Offline packed chat (HF chat template)|`packed_chat_mix`|Renders samples with the model's released HF chat template instead of `TEMPLATES`; requires `model.use_hf_chat_template: true` or `model.chat_template_path`.|

## 2. Input requirements (`data.wds_dir`)
The implementation reads uncompressed `*.tar` shards directly from `data.wds_dir`.
It does not unpack source shards into a flat directory.

Notes:

* `wds_pack.cli.scan_manifest` reads the message list from the field specified by `data.template_text_key`; it also accepts the common keys `messages` and `texts`.
* If the JSON files come from `tools/vlm_data_preprocess/convert_to_webdataset.py` (multi-scenario writes `texts` by default) you usually need to set `data.template_text_key` to `texts`.  
* `packed_multi_mix_qa`: JSON must declare `media`/`media_type` (`text`, `image`, or `video`). Image/video samples should supply `name`/`media_files`; if absent, media members are inferred from WDS parts by extension.
* `.tgz` input is not supported in V1 because efficient byte-range reads require uncompressed tar.

## 3. Quick start
```bash
cd tools/vlm_data_preprocess/offline_packing

# 1) Edit configs/config.yaml (or configs/config_256k.yaml for the 256k case)
# 2) Run the 4-step pipeline (reads configs/config.yaml by default)
bash scripts/pack_wds.sh
```

To switch to another config:

* Option 1: overwrite/copy it to `configs/config.yaml`
* Option 2: run each script manually with `--config your.yaml` (see next section)

## 4. Pipeline details (mirrors `pack_wds.sh`)

### Step 1: Scan WDS manifest and compute per-sample token length (`wds_pack.cli.scan_manifest`)
* Input: `*.tar` shards under `data.wds_dir`
* Process: read WDS samples directly from tar, render the chat text (the HF chat template when HF rendering is enabled; otherwise fall back to the handwritten `wds_pack.core.constants.TEMPLATES` picked by `sample.sample_type` + `model.model_type`), tokenise text+vision inputs with `AutoProcessor` or `AutoTokenizer`, and record tar byte locators
* Output: `{data.work_dir}/sample_manifest.sqlite` (authoritative manifest) and `skipped_overlong.jsonl` (samples whose `token_len > max_token_len`, always written); `sample_manifest.jsonl`, the combined `sample_len_report.txt`, the per-media token reports under `token_len/`, and `skipped_samples.jsonl` are kept only when `artifacts.debug_artifacts: true`

Manual run:
```bash
python -m wds_pack.cli.scan_manifest --config configs/config.yaml
```

### Step 2: Length bucketing & packing groups by media type (`wds_pack.cli.pack_bins`)
* Input: `sample_manifest.sqlite` (per-media token lengths)
* Process: pack samples into "boxes" separately for text/image/video under `sample.max_token_len`. The production algorithm is `packing.algorithm: best_fit_decreasing` (BFD); `hashbucket` is still available for the old exact-fill behaviour
* Output: `{data.work_dir}/bins/bins_plan_{text,image,video}.jsonl` (the intermediate `bins/bins_boxs_{text,image,video}.pkl` is kept only when `artifacts.keep_intermediate: true`)

Manual run:
```bash
python -m wds_pack.cli.pack_bins --config configs/config.yaml
```

### Step 3: Build pack plan (`wds_pack.cli.build_plan`)
* Input: per-media `bins/bins_plan_*.jsonl` + `sample_manifest.sqlite`
* Process: convert the per-media bins into stable packed sample plans
* Output: `{data.work_dir}/pack_plan.jsonl` (the `unpacked_samples.jsonl` diagnostic is kept only when `artifacts.debug_artifacts: true`)

Manual run:
```bash
python -m wds_pack.cli.build_plan --config configs/config.yaml
```

### Step 4: Write packed samples back to WebDataset (`wds_pack.cli.write_wds`)
* Input: `pack_plan.jsonl` + `sample_manifest.sqlite`; media bytes are read from source tar byte offsets
* Output: `data.packed_wds_dir/pretrain-*.tar` plus Energon meta (`.nv-meta/dataset.yaml` + tar indexes)

Manual run:
```bash
python -m wds_pack.cli.write_wds --config configs/config.yaml
```

## 5. Configuration (`configs/config.yaml`)
Key fields:

* `data.input_format` – set to `wds` for WDS-native packing
* `data.wds_dir` – input WebDataset directory containing uncompressed `*.tar` shards
* `data.template_text_key` – message field name in JSON (`messages` or `texts`)  
* `data.work_dir` – working directory for manifest, token reports, bins and pack plan
* `data.packed_wds_dir` – final packed WDS output directory  
* `sample.max_token_len` – target packing length (e.g. 8192 / 16384)  
* `sample.sample_type` – V1 supports `packed_multi_mix_qa` or `packed_chat_mix`
* `model.model_type` – model identifier used to pick the template  
* `model.processor_loader` – `auto_processor` for VLM processors, or `auto_tokenizer` for text-only smoke tests
* `model.processor_kwargs.*` – HF processor arguments passed to `transformers.AutoProcessor.from_pretrained`  
* `packed_wds.maxcount` / `maxsize` – tar-shard splitting strategy

Example (excerpt, full fields see `configs/config.yaml`):

```yaml
data:
  input_format: "wds"
  wds_dir: "/workspace/.../wds/"
  template_text_key: "texts"
  work_dir: "/workspace/.../packing_work/"
  packed_wds_dir: "/workspace/.../packed_wds/"

sample:
  max_token_len: 8192
  sample_type: packed_multi_mix_qa
```

## 6. Switching models / tuning image processing
Step 1’s token counts depend on the actual `AutoProcessor` logic, so you can change the model or image-preprocessing parameters via config:

* Change model: set `model.processor_kwargs.pretrained_model_name_or_path` to the desired HF model/processor; update `model.model_type` accordingly.  
* Adjust image-token budget / resolution: add processor-supported arguments under `model.processor_kwargs` (e.g. Qwen-VL’s `min_pixels`/`max_pixels`).  
* Template alignment: this is only needed when HF chat-template rendering is disabled (neither `model.use_hf_chat_template` nor `model.chat_template_path` is set) and the handwritten template is used as the fallback — if you add a new `model.model_type`, make sure `tools/vlm_data_preprocess/offline_packing/wds_pack/core/constants.py` contains the corresponding entry in `TEMPLATES[sample_type][model_type]`; otherwise Step 1 will raise “No template for sample_type=..., model_type=...”. When HF rendering is enabled this lookup is skipped and no `TEMPLATES` entry is required.
* Media pre-processing: under `media_preprocess` you can assign pre-processing function names per modality (implementations in `tools/vlm_data_preprocess/offline_packing/wds_pack/media/preprocess.py`) to control resize/crop/frame-reading behaviour.

## Acknowledgements

The WDS-native offline packing workflow in LoongForge is based on the multimodal
offline packing framework originally developed for LLaVA-OneVision-1.5 and later
migrated and upgraded for LLaVA-OneVision-2.

LoongForge previously collaborated with the LLaVA-OneVision work and has migrated
and adapted part of the LLaVA-OneVision offline packing capabilities.

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
