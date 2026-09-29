# LLM Data Preprocessing

Tokenize and index raw text/instruction corpora into the on-disk formats the
LoongForge/Megatron LLM training loop consumes. Two entry points:

| Script | Training phase | Output |
|--------|----------------|--------|
| `preprocess_pretrain_data.py` | Pretrain / continued pretrain | Megatron indexed dataset (`.bin` + `.idx`) |
| `preprocess_sft_data.py` | SFT (instruction tuning) | Tokenized HuggingFace `DatasetDict` directory |

## Prerequisites

Put Megatron and LoongForge on `PYTHONPATH` before launching (the `examples/`
scripts do this for you):

```bash
export MEGATRON_PATH=${MEGATRON_PATH:-/workspace/Loong-Megatron}
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-/workspace/LoongForge}
export PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH
```

A HuggingFace tokenizer (local directory or hub model id) is required via
`--hf-tokenizer-path` with `--tokenizer-type HFTokenizer` (recommended).

## Example scripts

Each model family under `examples/` ships ready-to-run wrappers you can copy and
adapt — just point the paths at your data and tokenizer:

- `examples/<model>/pretrain/preprocess_data.sh` — pretrain tokenization
- `examples/<model>/finetuning/preprocess_data.sh` — SFT tokenization

Both are provided for `deepseek_v2`, `deepseek_v3`, `llama2`, `llama3`,
`llama3.1`, `qwen`, `qwen1.5`, `qwen2`, `qwen2.5`, and `qwen3`. They differ only
in tokenizer path, `--seq-length`, and `--chat-template`, so any one is a good
starting point for a new model.

## 1. Pretrain data

**Input:** a JSON Lines file, one document per line, with the text under a JSON
key (default `text`):

```json
{"text": "The quick brown fox ..."}
{"text": "Another document ..."}
```

**Run** (from `examples/llama3/pretrain/preprocess_data.sh`):

```bash
PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
    python ${LOONGFORGE_PATH}/tools/llm_data_preprocess/preprocess_pretrain_data.py \
        --input       /path/to/train.jsonl \
        --output-prefix /path/to/output/pile-llama \
        --tokenizer-type   HFTokenizer \
        --hf-tokenizer-path /path/to/hf/tokenizer \
        --json-keys   text \
        --workers     50 \
        --append-eod
```

**Output:** `<output-prefix>_text_document.bin` and `.idx` — pass the prefix as a
training data path.

**Key arguments:**

| Argument | Required | Description |
|----------|:---:|-------------|
| `--input` | ✅ | Path to the input `.jsonl` |
| `--output-prefix` | ✅ | Output path prefix (no suffix) for the `.bin`/`.idx` pair |
| `--tokenizer-type` | ✅ | `HFTokenizer` (recommended), `Llama2Tokenizer`, or `NullTokenizer` |
| `--hf-tokenizer-path` | – | HF tokenizer dir or hub id (with `HFTokenizer`) |
| `--json-keys` | – | JSON key(s) to extract, space-separated (default `text`) |
| `--append-eod` | – | Append an end-of-document token per record |
| `--workers` | ✅ | Worker processes; aim for `workers * partitions ≈ CPU cores` |
| `--partitions` | – | Split the input into N shards for parallel processing (default 1) |
| `--model-family` | – | Model family (required for Qwen; see `--help` for choices) |

## 2. SFT data

**Input:** an instruction dataset in `alpaca`, `sharegpt`, or `openai` format.
The column/tag mapping is described by a dataset config (see
`configs/data/sft_dataset_config.yaml`); the built-in `default` entry expects
alpaca-style `instruction` / `input` / `output` columns.

**Run** (from `examples/llama3/finetuning/preprocess_data.sh`):

```bash
PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
    python ${LOONGFORGE_PATH}/tools/llm_data_preprocess/preprocess_sft_data.py \
        --input        /path/to/sft_data.json \
        --output-path  /path/to/output/sft_tokenized \
        --seq-length   2048 \
        --chat-template llama3 \
        --tokenizer-type   HFTokenizer \
        --hf-tokenizer-path /path/to/hf/tokenizer \
        --workers      50 \
        --split        100,0,0
```

**Output:** a tokenized dataset directory (train/valid/test per `--split`).

**Key arguments:**

| Argument | Required | Description |
|----------|:---:|-------------|
| `--input` | ✅ | Path to the input instruction JSON |
| `--output-path` | ✅ | Output directory for the tokenized dataset |
| `--chat-template` | ✅ | Prompt template to apply, e.g. `llama3`, `qwen`, `deepseek` (see below) |
| `--hf-tokenizer-path` | ✅ | HF tokenizer dir or hub id |
| `--seq-length` | – | Max sequence length; longer samples are truncated (or discarded, see below) |
| `--split` | – | Train,valid,test proportions (default `100,0,0`) |
| `--sft-dataset-config` | – | YAML mapping of dataset format/columns (default: built-in config) |
| `--sft-dataset` | – | Named entry within the dataset config (default `default`) |
| `--packing-sft-data` | – | Pack multiple samples into one sequence to reduce padding |
| `--packing-buffer-size` | – | Batch size for packing (default 10000) |
| `--context-parallel-size` | – | Set to the training CP size so packed data is padded correctly |
| `--train-on-prompt` | – | Also compute loss on the prompt (default: response only) |
| `--history-mask-loss` | – | Compute loss only on the last-turn response |
| `--eod-mask-loss` | – | Mask loss on end-of-document tokens |
| `--enable-discard-sample` | – | Drop samples longer than `--seq-length` instead of truncating |
| `--workers` | ✅ | Worker processes |

### Supported chat templates

`--chat-template` accepts any template registered in
`loongforge/data/chat_template.py`, including `llama2` / `llama3` / `llama3.1`,
`qwen` / `qwen2.5-hf` / `qwen3-hf`, `deepseek` / `deepseek3` / `deepseek4`,
`glm5`, `minimax-m2`, `mistral`, `alpaca`, and more. Run the script with
`--help` (its `choices` list is the authoritative set) to see everything
available in your checkout.

## Full option reference

Every flag with its help text:

```bash
python ${LOONGFORGE_PATH}/tools/llm_data_preprocess/preprocess_pretrain_data.py --help
python ${LOONGFORGE_PATH}/tools/llm_data_preprocess/preprocess_sft_data.py --help
```
