#!/bin/bash
# =============================================================================
# convert2wds.sh -- ERNIE offline JSONL -> WebDataset conversion script
#
# Flow:  Original format (Step 1) -> sharegpt format (Step 2) -> WebDataset tar
#
# ---- Original data format (ERNIE offline pipeline, text_info + image_info) ----
#
#   {
#     "text_info": [
#       {"text": "User question 1",   "tag": "mask"},       // tag=mask    -> user (excluded from loss)
#       {"text": "Assistant answer 1", "tag": "no_mask"},    // tag=no_mask -> assistant (included in loss)
#       {"text": "User question 2",   "tag": "mask"},
#       {"text": "Assistant answer 2", "tag": "no_mask"},
#       ...
#     ],
#     "image_info": [
#       {
#         "image_url": "./DoclingMatix/test.jpg",      // Relative image path
#         "matched_text_index": 0                       // Image inserted before text_info[0]
#       }
#     ]
#   }
#
# ---- Target data format (sharegpt, consumed by convert_to_webdataset.py) ----
#
#   {
#     "messages": [
#       {"role": "user",      "content": "<image>\nUser question 1"},   // <image> generated from image_info
#       {"role": "assistant", "content": "Assistant answer 1"},
#       {"role": "user",      "content": "User question 2"},
#       {"role": "assistant", "content": "Assistant answer 2"}
#     ],
#     "image": "DoclingMatix/test.jpg"       // Single image uses "image"; multiple images use "images": [...]
#   }
#
# ---- Final WebDataset tar internal structure ----
#
#   image_0.json                  ->  {"texts": [messages...], "media": "image", "name": ["0_test.jpg"]}
#   image_0.0_test.jpg            ->  Image binary
#
#   .nv-meta/dataset.yaml         ->  subflavors: {sample_type: ernie_mix_qa}
#
# =============================================================================
set -e

omni=/workspace/LoongForge
input_jsonl=/workspace/dataset/ERNIE/examples/data/sft_vl-demo.jsonl
converted_jsonl=/workspace/dataset/ERNIE/examples/data/sft_vl-demo_sharegpt.jsonl
image_dir=/workspace/dataset/ERNIE/examples/data/
output_wds=/workspace/dataset/wds

# ---- Step 1: Convert ERNIE offline format (text_info+image_info) to sharegpt (messages+image) ----
echo "Converting ERNIE format -> sharegpt format ..."
python3 -c "
import json, sys

def convert_entry(entry):
    text_info = entry['text_info']
    image_info = entry.get('image_info', [])

    # Map image_info indices to paths
    image_at_index = {}
    all_image_paths = []
    for img in image_info:
        idx = img['matched_text_index']
        path = img['image_url'].lstrip('./')
        image_at_index.setdefault(idx, []).append(path)
        all_image_paths.append(path)

    # Pair (mask, no_mask) -> (user, assistant) turns
    messages = []
    i = 0
    while i < len(text_info):
        user_text = text_info[i]['text']
        if i in image_at_index:
            prefix = '<image>' * len(image_at_index[i])
            user_text = prefix + user_text
        messages.append({'role': 'user', 'content': user_text})
        if i + 1 < len(text_info):
            messages.append({'role': 'assistant', 'content': text_info[i + 1]['text']})
        i += 2

    out = {'messages': messages}
    if len(all_image_paths) == 1:
        out['image'] = all_image_paths[0]
    elif len(all_image_paths) > 1:
        out['images'] = all_image_paths
    return out

count = 0
with open('${input_jsonl}') as fin, open('${converted_jsonl}', 'w') as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        out = convert_entry(json.loads(line))
        fout.write(json.dumps(out, ensure_ascii=False) + '\n')
        count += 1
print(f'Converted {count} samples -> ${converted_jsonl}')
"

# ---- Step 2: Convert sharegpt JSONL to WebDataset ----
echo "Converting sharegpt JSONL -> WebDataset ..."
python $omni/tools/data_preprocess/vlm/convert_to_webdataset.py \
  --output_dir $output_wds \
  --json_file $converted_jsonl \
  --image_dir $image_dir \
  --media image \
  --columns_messages messages \
  --sample_type ernie_mix_qa

echo "Done. WebDataset at: $output_wds"
