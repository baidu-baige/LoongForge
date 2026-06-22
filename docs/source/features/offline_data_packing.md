# Offline Packing  
This module provides an “offline sequence-packing” pipeline: it reads source **WebDataset tar shards** directly, groups and re-orders the samples according to `max_token_len`, and finally produces a **packed WebDataset** (`pretrain-*.tar` plus Energon meta files).
By concatenating variable-length sequences up to the target length we reduce padding and increase training throughput.

Entry script:  
`tools/data_preprocess/vlm/offline_packing/scripts/pack_wds.sh` (4 steps, see below).

## 1. Supported packing scenarios (`sample.sample_type`)

We currently support packing for single-sample captioning, VQA, and multi-modal mixed-QA formats.

|Scenario|`sample_type`|Description|
|---|---|---|
|Offline packed image/video/text mixed QA|`packed_multi_mix_qa`|Input WDS JSON must declare `media`/`media_type`; packs are homogeneous by media type.|

## 2. Input requirements (`data.wds_dir`)
The implementation reads uncompressed `*.tar` shards directly from `data.wds_dir`.
It does not unpack source shards into a flat directory.

Notes:

* `scan_wds_manifest.py` reads the message list from the field specified by `data.template_text_key`; it also accepts the common keys `messages` and `texts`.
* If the JSON files come from `tools/data_preprocess/vlm/convert_to_webdataset.py` (multi-scenario writes `texts` by default) you usually need to set `data.template_text_key` to `texts`.  
* `packed_multi_mix_qa`: JSON must declare `media`/`media_type` (`text`, `image`, or `video`). Image/video samples should supply `name`/`media_files`; if absent, media members are inferred from WDS parts by extension.
* `.tgz` input is not supported in V1 because efficient byte-range reads require uncompressed tar.

## 3. Quick start
```bash
cd tools/data_preprocess/vlm/offline_packing

# 1) Edit config.yaml (or copy packed_vqa_demo.yaml)
# 2) Run the 4-step pipeline (reads config.yaml by default)
bash scripts/pack_wds.sh
```

To switch to another config:

* Option 1: overwrite/copy it to `config.yaml`  
* Option 2: run each script manually with `--config your.yaml` (see next section)

## 4. Pipeline details (mirrors `pack_wds.sh`)

### Step 1: Scan WDS manifest and compute per-sample token length (`scan_wds_manifest.py`)
* Input: `*.tar` shards under `data.wds_dir`
* Process: read WDS samples directly from tar, pick the template (`utils.TEMPLATES`) according to `sample.sample_type` + `model.model_type`, tokenise text+vision inputs with `AutoProcessor` or `AutoTokenizer`, and record tar byte locators
* Output: `{data.work_dir}/sample_manifest.sqlite`, `{data.work_dir}/sample_manifest.jsonl`, and per-media token reports

Manual run:
```bash
python scan_wds_manifest.py --config config.yaml
```

### Step 2: Length bucketing & packing groups by media type (`do_hashbacket.py`)
* Input: `token_len/sample_len_report_{text,image,video}.txt`
* Process: build hash buckets separately for text/image/video, pack samples into “boxes” under `sample.max_token_len`
* Output: `{data.work_dir}/bins/bins_boxs_{text,image,video}.pkl`

Manual run:
```bash
python do_hashbacket.py --config config.yaml
```

### Step 3: Generate pack plan (`build_pack_plan.py`)
* Input: per-media `bins_boxs_*.pkl` + `sample_manifest.sqlite`
* Process: convert hashbucket boxes into stable packed sample plans
* Output: `{data.work_dir}/pack_plan.jsonl`

Manual run:
```bash
python build_pack_plan.py --config config.yaml
```

### Step 4: Write packed samples back to WebDataset (`packed_to_wds.py`)
* Input: `pack_plan.jsonl` + `sample_manifest.sqlite`; media bytes are read from source tar byte offsets
* Output: `data.packed_wds_dir/pretrain-*.tar` plus Energon meta (`.nv-meta/dataset.yaml` + tar indexes)

Manual run:
```bash
python packed_to_wds.py --config config.yaml
```

## 5. Configuration (`config.yaml`)
Key fields:

* `data.input_format` – set to `wds` for WDS-native packing
* `data.wds_dir` – input WebDataset directory containing uncompressed `*.tar` shards
* `data.template_text_key` – message field name in JSON (`messages` or `texts`)  
* `data.work_dir` – working directory for manifest, token reports, bins and pack plan
* `data.packed_wds_dir` – final packed WDS output directory  
* `sample.max_token_len` – target packing length (e.g. 8192 / 16384)  
* `sample.sample_type` – V1 supports `packed_multi_mix_qa`
* `model.model_type` – model identifier used to pick the template  
* `model.processor_loader` – `auto_processor` for VLM processors, or `auto_tokenizer` for text-only smoke tests
* `model.processor_kwargs.*` – HF processor arguments passed to `transformers.AutoProcessor.from_pretrained`  
* `packed_wds.maxcount` / `maxsize` – tar-shard splitting strategy

Example (excerpt, full fields see `config.yaml`):

```yaml
data:
  input_format: "wds"
  wds_dir: "/mnt/cluster/.../wds/"
  template_text_key: "texts"
  work_dir: "/mnt/cluster/.../packing_work/"
  packed_wds_dir: "/mnt/cluster/.../packed_wds/"

sample:
  max_token_len: 8192
  sample_type: packed_multi_mix_qa
```

## 6. Switching models / tuning image processing
Step 1’s token counts depend on the actual `AutoProcessor` logic, so you can change the model or image-preprocessing parameters via config:

* Change model: set `model.processor_kwargs.pretrained_model_name_or_path` to the desired HF model/processor; update `model.model_type` accordingly.  
* Adjust image-token budget / resolution: add processor-supported arguments under `model.processor_kwargs` (e.g. Qwen-VL’s `min_pixels`/`max_pixels`).  
* Template alignment: if you add a new `model.model_type`, make sure `tools/data_preprocess/vlm/offline_packing/utils.py` contains the corresponding entry in `TEMPLATES[sample_type][model_type]`; otherwise Step 1 will raise “No template found for model_type ...”.  
* Media pre-processing: under `media_preprocess` you can assign pre-processing function names per modality (implementations in `tools/data_preprocess/vlm/offline_packing/media_preprocess_utils.py`) to control resize/crop/frame-reading behaviour.

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
