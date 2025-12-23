# Optional Configs 功能说明

本功能支持灵活选择要回归测试的模型配置，既可以使用 `configs/` 目录下的固定配置，也可以使用 `optional_configs/` 目录下的可选配置。

## 目录结构

```
tests/
├── configs/                    # 固定的回归测试配置（默认加载）
│   ├── common.yaml
│   ├── llama2_7b.yaml
│   ├── qwen2.5_vl_7b.yaml
│   └── ...
├── optional_configs/           # 可选的测试配置（按需加载）
│   ├── qwen2_5_0_5b.yaml
│   ├── qwen2_5_72b.yaml
│   ├── llama3_70b.yaml
│   └── ...
└── tools/
    └── generate_optional_configs.py  # 配置生成脚本
```

## 使用方式

### 1. 基本使用（只使用 configs/ 目录）

```bash
# 运行默认配置中的所有模型
python main.py

# 运行指定模型
python main.py --models llama2_7b qwen2.5_vl_7b
```

### 2. 包含可选配置

```bash
# 方式1: 使用 --include_optional 参数
python main.py --include_optional --models qwen2_5_72b

# 方式2: 使用 --extra_configs_dirs 指定额外目录
python main.py --extra_configs_dirs optional_configs custom_configs --models llama3_70b
```

### 3. 使用模式匹配

```bash
# 只运行匹配 "qwen*" 模式的模型
python main.py --include_optional --include_patterns "qwen*"

# 运行所有模型，但排除 72b 的大模型
python main.py --include_optional --exclude_patterns "*_72b"

# 组合使用
python main.py --include_optional --include_patterns "qwen2.5*" --exclude_patterns "*_72b"
```

### 4. 列出所有可用模型

```bash
python main.py --list_available_models
```

## 命令行参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `--models` | 指定要测试的模型名称 | `--models llama2_7b qwen2.5_7b` |
| `--include_optional` | 包含 optional_configs 目录 | `--include_optional` |
| `--extra_configs_dirs` | 额外的配置目录 | `--extra_configs_dirs dir1 dir2` |
| `--include_patterns` | 只包含匹配的模型（支持通配符） | `--include_patterns "qwen*" "llama3_*"` |
| `--exclude_patterns` | 排除匹配的模型（支持通配符） | `--exclude_patterns "*_72b" "deepseek*"` |
| `--list_available_models` | 列出所有可用模型并退出 | `--list_available_models` |

## 通配符说明

- `*` 匹配任意字符（0个或多个）
- `?` 匹配单个字符

示例：
- `qwen*` 匹配 qwen2_5_7b, qwen3_14b 等
- `*_7b` 匹配 llama2_7b, qwen2_5_7b 等
- `qwen2_5_?b` 匹配 qwen2_5_3b, qwen2_5_7b 等

## 生成可选配置

从 `examples/` 目录自动生成可选配置：

```bash
cd tests

# 生成所有模型的配置
python tools/generate_optional_configs.py

# 只生成特定模型家族的配置
python tools/generate_optional_configs.py --models qwen2.5 llama3

# 预览模式（不实际生成文件）
python tools/generate_optional_configs.py --dry_run
```

## main_start.sh 配置

在 `main_start.sh` 中配置要测试的模型：

```bash
# ============================================================================
# 模型选择配置
# ============================================================================

# 方式1: 直接指定模型名称
model_names="llavaov_1.5_4b llama2_7b"

# 方式2: 使用模式匹配
include_patterns="qwen*"
exclude_patterns="*_72b"

# 方式3: 是否包含 optional_configs 目录
include_optional=true

# 额外的配置目录
extra_configs_dirs="optional_configs"
```

## 工作流程示例

### 场景1: 日常回归测试

只使用 `configs/` 下的固定配置：

```bash
./main_start.sh
```

### 场景2: 新模型上线前的全量测试

1. 生成所有模型的可选配置：
```bash
python tools/generate_optional_configs.py
```

2. 运行全量测试：
```bash
python main.py --include_optional --include_patterns "qwen2.5*"
```

### 场景3: 测试特定尺寸的模型

```bash
# 测试所有 7B 模型
python main.py --include_optional --include_patterns "*_7b"

# 测试所有小于 14B 的模型
python main.py --include_optional --include_patterns "*_0_5b" "*_1_5b" "*_3b" "*_7b"
```

## 注意事项

1. `optional_configs/` 下的配置是模板配置，需要根据实际情况调整路径等参数
2. 模式匹配会应用于所有配置目录中的模型
3. `--include_patterns` 和 `--exclude_patterns` 可以同时使用，先应用包含规则，再应用排除规则
4. 如果同时指定了 `--models` 和模式匹配参数，以模式匹配的结果为准
