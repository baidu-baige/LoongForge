# AGENTS.md

LoongForge：基于 Loong-Megatron（Megatron-LM 补丁分支）和 TransformerEngine 的大规模训练框架，支持 LLM、VLM、VLA（具身）和扩散模型，运行在 NVIDIA GPU 与昆仑 XPU 上。训练阶段：pretrain、SFT。

## 环境

```bash
git clone --recurse-submodules https://github.com/baidu-baige/LoongForge.git
uv pip install -e ".[gpu]"            # NVIDIA GPU；昆仑 XPU 用 ".[xpu]"
python setup_env.py --te-tag v2.9     # 仅构建 TransformerEngine（打 patches/TransformerEngine_v2.9）
```

- Loong-Megatron 是 `third_party/Loong-Megatron` 子模块，不由 `setup_env.py` 处理。
- Docker 构建：`docker build --build-arg COMPILE_ENV=<ampere|hopper|blackwell> -f docker/Dockerfile .`
- 打包：`python -m build --sdist --wheel --outdir dist/`

## 常用命令

```bash
pre-commit run --all-files            # 空白、YAML、SPDX 头；ruff 配置见 pyproject.toml（行宽 120）
ruff check loongforge tools tests
```

训练启动，`PYTHONPATH` 必须同时含 Megatron 和 LoongForge：

```bash
PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
  torchrun --nproc_per_node 8 --nnodes $NNODES \
  $LOONGFORGE_PATH/loongforge/train.py \
  --model-name <name> --training-phase pretrain|sft ...
```

`--model-name` 经 `loongforge/models/catalog.py` 映射到 engine 与配置；也可用 `--config-file` 直接指定 YAML。每个模型族的脚本在 `examples/<model>/{pretrain,sft,checkpoint_convert}/`，XPU 版在 `examples_xpu/`。

## 测试

不使用 pytest 跑 E2E；E2E 是 YAML 驱动的自研框架。入口不会下载数据、HF 模型和转换后的检查点，需先备好。

```bash
bash tests/llm_vlm/main_start.sh      # 默认 CI 套件：tests/llm_vlm/configs/
```

- 跑单个模型：改 `tests/llm_vlm/main_start.sh` 里的 `model_names`；`optional_configs/` 下的模型需同时设 `include_optional=true`，整个系列用 `model_names="NONE"` + `optional_subdir=<series>`。
- 具身/VLA 回归：`bash tests/embodied/run.sh --chip <a|...> [--models ...] [--dry_run] [--list_models]`。
  - 新训练脚本必须登记到 `tests/embodied/config/scripts.yaml`（路径相对 `examples/`），并为每个支持的芯片添加 `tests/embodied/baseline/<chip>/<name>.json`。
  - 训练脚本的数据、检查点、缓存、输出路径用环境变量覆盖，不靠执行器改写参数。
  - loss 和 `grad_norm` 硬检查；基线缺失、非零退出、NaN/Inf 均判失败；性能退化只告警。
  - 基线按芯片区分，未验证不要跨硬件复用。
  - 改动视频/动作预处理、text-embedding 缓存或分布式策略时，先 `--dry_run`，再跑真实回归。

## 架构约束

分层：`train → catalog/parser/training.registry → runner → method + data + models + engine`。

- **models** 不导入 training、runner、DataLoader，不读取 engine 全局训练状态。
- **data** 第一层只有 4 个家族目录（`llm/ vlm/ embodied/ diffusion/`）、4 个 `<family>_dataloader.py` 和只含 docstring 的 `__init__.py`；DataLoader、sampler 只在入口组装；只可用 `engines.mcore.tokenizer`（`llm_dataloader.py` 另可用 `chunkpipe`），不读 engine 全局状态，不创建进程组。
- **`chat_templates/`、`constants.py`** 不导入 data 和 engines；`constants.py` 只用标准库。
- **engine** 不扫描 method，不按模型名复制训练循环。
- catalog 和配置在导入阶段不加载未选中的 engine、权重或可选依赖。
- `engines/mcore/` 与 `engines/torch/` 的同名文件是同一角色的两种实现。

## 禁止事项

1. 不新建 `utils/` 或 `custom/` 目录。已有 `*_utils.py` 保留，但不作为新功能的默认落点。
2. 不 fork 上游函数：Loong-Megatron 已有的直接 import；需要扩展点时在上游加 hook。
3. 不在 `__init__.py` 中扫描导入全部模型：只导入选中的模块，依赖缺失时报出具体模块名。
4. models/data/chat_templates 不读 engine 全局状态（`get_args`、`get_tokenizer`、`get_model_config`、`get_chat_template`），由 method 传参。
5. 不提交内部主机名、内部镜像源、个人家目录路径；发布前用 `sensitive-scan` skill 扫描。

## Skills

`skills/`（`.claude/skills` 是其符号链接）：`sensitive-scan`、`submit-pr`、`code-review`。开 PR 用 `submit-pr`（含 PR 标题的 module/type 校验）。

## Code Review

评审 PR 或 diff 时严格遵循 `skills/code-review/SKILL.md`：检查清单、严重度标签（🔴 Critical / 🟠 Major / 🟡 Minor / 🟢 Nit）和输出格式都以它为准。

- 与具体代码相关的问题用 `mcp__github_inline_comment__create_inline_comment` 发在精确的 `file:line`（或区间）上，不要堆在总结里。
- 总结评论只放 `Verdict`、2–4 句 `Summary`、`Tests` 行和 `Checklist` 表。
- 只对真实评审发现设 `confirmed: true`，探测和自测不设。
