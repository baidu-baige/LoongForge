# Offline Packing  
This module provides an “offline sequence-packing” pipeline: it takes a **sample-level** directory (one `*.json` plus its media files per sample), groups and re-orders the samples according to `max_token_len`, and finally produces a **packed WebDataset** (`pretrain-*.tar` plus Energon meta files).  
By concatenating variable-length sequences up to the target length we reduce padding and increase training throughput.

Entry script:  
`tools/data_preprocess/vlm/offline_packing/scripts/pack_wds.sh` (4 steps, see below).

## 1. Supported packing scenarios (`sample.sample_type`)

We currently support packing for single-sample captioning, VQA, and multi-modal mixed-QA formats.

|Scenario|`sample_type`|Description|
|---|---|---|
|Offline packed caption|`packed_captioning`|Produces a `CrudeWebdataset`; downstream code must parse this `sample_type`.|
|Offline packed single-image QA|`packed_vqa`|Same as above.|
|Offline packed image+video mixed QA|`packed_multi_mix_qa`|Same as above (input JSON must declare media types and file lists).|

## 2. Input requirements (`data.wds_dir`)
The implementation **does NOT read tar shards directly**; it expects a flat, random-accessible directory:

* Many `*.json` files (each file = one sample / one WDS json payload).  
* Media files (images/videos) sitting in the same directory, or referenced via a relative path resolvable from that directory.

If your data are already `pretrain-*.tar` shards produced by `convert_to_webdataset.py`, unpack them first:

```bash
mkdir -p /path/to/wds_flat
for t in /path/to/wds/pretrain-*.tar; do tar -xf "$t" -C /path/to/wds_flat; done
```

Notes:

* `get_sample_len.py` reads the message list from the field specified by `data.template_text_key`; it also accepts the common keys `messages` and `texts`.  
* If the JSON files come from `tools/data_preprocess/vlm/convert_to_webdataset.py` (multi-scenario writes `texts` by default) you usually need to set `data.template_text_key` to `texts`.  
* `packed_vqa` / `packed_captioning`: if the JSON does not contain an explicit `media_files/name` field, the code tries to find a media file with the same stem (e.g. `0001.json` → `0001.jpg`).  
* `packed_multi_mix_qa`: JSON must declare `media`/`media_type` (`image` or `video`) and supply `name`/`media_files` list (nested lists allowed).

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

### Step 1: Compute per-sample token length (`get_sample_len.py`)
* Input: `*.json` + media files under `data.wds_dir`  
* Process: pick the template (`utils.TEMPLATES`) according to `sample.sample_type` + `model.model_type`, tokenise text+vision inputs with `AutoProcessor`, record token length for every sample  
* Output: `{data.wds_dir}/.temp/sample_len_report.txt` (`sample_id: token_len`)

Manual run:
```bash
python get_sample_len.py --config config.yaml
```

### Step 2: Length bucketing & packing groups (`do_hashbacket.py`)
* Input: `sample_len_report.txt`  
* Process: build hash buckets, pack samples into “boxes” under `sample.max_token_len`  
* Output: `{data.packed_json_dir}/bins_boxs.pkl` (each box = list of sample ids that will be concatenated into one packed sample)

Manual run:
```bash
python do_hashbacket.py --config config.yaml
```

### Step 3: Generate packed intermediate JSON (`prepare_raw_samples.py`)
* Input: `bins_boxs.pkl` + original `*.json`/media  
* Process: aggregate samples per box, produce packed-json with fields such as `prompts`/`captions`/`media_files`/`media_type`  
* Output: `{data.packed_json_dir}/row_packing_jsons/*.json`

Manual run:
```bash
python prepare_raw_samples.py --config config.yaml
```

### Step 4: Write packed JSON back to WebDataset (`packed_to_wds.py`)
* Input: `row_packing_jsons/*.json` + media (looked up under `{data.wds_dir}` or `{data.packed_json_dir}/row_packing_images`)  
* Output: `data.packed_wds_dir/pretrain-*.tar` (or `{data.packed_json_dir}/packed_wds` if not configured) plus Energon meta (`.wds/dataset.yaml` + index)

Manual run:
```bash
python packed_to_wds.py --config config.yaml
```

## 5. Configuration (`config.yaml`)
Key fields:

* `data.wds_dir` – input sample directory (`*.json` + media)  
* `data.template_text_key` – message field name in JSON (`messages` or `texts`)  
* `data.packed_json_dir` – working directory for intermediate pkl/json  
* `data.packed_wds_dir` – final packed WDS output directory  
* `sample.max_token_len` – target packing length (e.g. 8192 / 16384)  
* `sample.sample_type` – see Section 1  
* `model.model_type` – model identifier used to pick the template  
* `model.processor_kwargs.*` – HF processor arguments passed to `transformers.AutoProcessor.from_pretrained`  
* `packed_wds.maxcount` / `maxsize` – tar-shard splitting strategy

Example (excerpt, full fields see `config.yaml`):

```yaml
data:
  wds_dir: "/mnt/cluster/.../wds_flat/"
  template_text_key: "messages"
  packed_json_dir: "/mnt/cluster/.../packed_json/"
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