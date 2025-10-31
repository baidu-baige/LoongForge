## 配置文件使用说明
在指令微调场景中，训练数据主要由对话数据组成，而对话数据通常由对话文本和对话标签组成，而不同数据集的字段格式可能也是多样化。为了便于系统进行数据集读取，需要提供一个配置文件，该配置文件描述了对话数据集的格式信息。

* `configs/dataset_config.json` 是该配置文件的一个模板示例（目前仅支持 json 文件），用户可以移动或修改文件配置，AIAK 期望用户进行 SFT 训练时通过 `--sft-dataset-config` 传递配置文件路径（注：预训练不需要）。
* 如果用户没有通过 `--sft-dataset-config` 提供配置文件路径，系统将默认查找 `configs/dataset_config.json` 作为配置文件用于后续数据集解析。


## 数据集格式和使用说明
`configs/dataset_config.json` 的格式规范，主要参考了开源社区的常用做法，参见 `https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md`。

目前 AIAK 仅支持 `alpaca` 数据集格式，`sharegpt` 数据集格式后续会增加支持。

在配置文件中，数据集配置格式如下：
```
{
    "custom_dataset_name（用户可自定义）": {
        "format": "数据集格式（可选，取值范围: alpaca，默认: alpaca）",
        "columns（可选，用户如未指定则按照如下默认值配置，用户如若指定则会严格按照指定的字段项进行解析，不会使用任何默认值）": {
            "prompt": "数据集中代表提示词的字段名称（用户可自定义，默认: instruction）",
            "query": "数据集中代表输入的字段名称（用户可自定义，默认: input）",
            "response": "数据集中代表回答的字段名称（用户可自定义，默认: output）",
            "system": "数据集中代表系统提示的字段名称（用户可自定义，默认: None）",
            "history": "用于多轮对话，数据集中代表历史对话的字段名称（用户可自定义，默认: None）",
        }
    },
    ...
}
```

训练使用方式：
* 当用户使用自定义数据集训练时，用户首先需要在给定的配置文件中添加自定义数据集的名称，并指定其格式和各字段名称。
* 在训练启动时，通过 `--sft-dataset` 或 `--sft-train-dataset`/`--sft-valid-dataset`/`--sft-test-dataset` 等参数指定数据集名称。
* 传递的数据集名称，需和 `--data-path` 或 `--train-data-path`/`--valid-data-path`/`--test-data-path` 指定的数据集文件一一对应，如果指定多个数据集文件，系统会按照指定的顺序依次进行解析。

其他使用说明：
* 如果用户没有通过上述 `--sft-*—dataset` 参数指定数据集名称，系统将默认使用 `configs/dataset_config.json` 中的 `default` 项作为训练数据集的默认格式进行解析。
* 如果用户需要指定多个数据集文件，而每个数据集的格式完全相同，也可以仅传递一个数据集名称，系统会按照指定的数据集名称对应的字段格式解析所有数据集文件。


## 数据集格式举例
### Alpaca 格式
#### SFT 数据

假设训练数据集文件内容形式如下（注：数据集的每条样本，都应该包括相同的字段项，部分字段项内容可以为空）:
```
[
  {
    "instruction": "用户指令",
    "input": "问题输入",
    "output": "模型的回答",
    "system": "系统提示词",
    "history": [
      ["第一轮对话的用户指令", "第一轮对话的模型回答"],
      ["第二轮对话的用户指令", "第二轮对话的模型回答"]
    ]
  }
]
```

对应的配置方式如下：
```
"custom_dataset_name": {
  "format": "alpaca",
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output",
    "system": "system",
    "history": "history"
  }
}

```