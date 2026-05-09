# 许可证与文件头指南

本文档描述如何在 LoongForge 中添加版权和许可证头。LoongForge 仓库基于 [Apache License 2.0](https://github.com/baidu-baige/LoongForge/blob/master/LICENSE) 发布。

同时，本仓库中的部分文件衍生自第三方开源项目。这些文件必须继续遵循其原始版权和许可证要求。

在实践中，贡献者应遵循以下原则：
- 为 LoongForge 编写的原创文件应使用简短的基于 SPDX 的 Apache-2.0 头。
- 衍生自第三方项目的文件必须原样保留上游版权和许可证声明。
- 修改第三方衍生文件时，贡献者必须添加明确的修改和来源声明。

---

## 情况 1：为 LoongForge 编写的原创文件

对于由 LoongForge 团队或贡献者编写的新源文件，使用简短的基于 SPDX 的 Apache-2.0 头。这使文件保持简洁，同时保持清晰、机器可读且与现代工具一致。

### Python
```python
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
```

### Shell
```bash
#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
```

### C / C++ / CUDA
```cpp
// Copyright 2026 The LoongForge Authors.
// SPDX-License-Identifier: Apache-2.0
```

---

## 情况 2：衍生自第三方项目的文件

对于改编自第三方项目的文件（无论是 Apache 2.0、MIT、BSD 或其他许可证），我们使用**通用极简模板**。

**黄金法则：** 不要修改上游作者的原始头部。在前面添加 LoongForge 版权、SPDX 标识符、一行来源和许可证声明，然后*原样*粘贴上游头部（无论其是 1 行还是 20 行）。

### 通用模板
```python
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from [UpstreamProject Name] under the [License Name, e.g., MIT / Apache-2.0] License.
# [👇 在此原样粘贴源文件中的所有原始上游头部注释]
```

### 实际示例 A：上游有非常短的头部
如果上游文件只有一行版权声明，只需保留那一行。

```python
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Megatron-LM under the BSD 3-Clause License.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

# 你的代码从这里开始...
```

### 实际示例 B：上游有长的样板头部
如果上游文件有很长的许可证文本，请不做修改地粘贴整个块。

```python
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from ERNIE.
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 你的代码从这里开始...
```

---

## 推荐决策规则

在决定使用哪种头部时，遵循以下简单逻辑：
1. **是否为原创？** 如果是，使用简短的 SPDX 头（情况 1）。
2. **是否修改自第三方？** 使用通用模板（情况 2）：添加我们的 3 行 LoongForge 头 + 来源声明，然后在下方原样复制粘贴原始作者的头部。
