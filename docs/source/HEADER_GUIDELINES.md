# License and File Header Guidelines

This document describes how to add copyright and license headers in LoongForge. The LoongForge repository is released under the [Apache License 2.0](https://github.com/baidu-baige/LoongForge/blob/master/LICENSE).

At the same time, some files in this repository are derived from third-party open-source projects. Those files must remain subject to their original copyright and license requirements. 

In practice, contributors should follow these principles:
- Original files written for LoongForge should use a short SPDX-based Apache-2.0 header.
- Files derived from third-party projects must retain upstream copyright and license notices verbatim.
- When modifying third-party derived files, contributors must add a clear modification and origin notice.

---

## Case 1: Original Files Written for LoongForge

For new source files authored by the LoongForge team or contributors, use the short SPDX-based Apache-2.0 header. This keeps files concise while remaining clear, machine-readable, and consistent with modern tooling.

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

## Case 2: Files Derived from Third-Party Projects

For files adapted from third-party projects (whether Apache 2.0, MIT, BSD, or others), we use a **Universal Minimalist Template**.

**The Golden Rule:** Do not alter the upstream author's original header. Prepend the LoongForge copyright, an SPDX identifier, a single line stating the origin and license, and then paste the upstream header *exactly as it is* (whether it is 1 line or 20 lines).

### The Universal Template
```python
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from [UpstreamProject Name] under the [License Name, e.g., MIT / Apache-2.0] License.
# [👇 Paste all original upstream header comments here exactly as they appear in the source]
```

### Real-World Example A: Upstream has a very short header
If the upstream file only has a single copyright line, just keep that single line.

```python
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Megatron-LM under the BSD 3-Clause License.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

# Your code begins here...
```

### Real-World Example B: Upstream has a long boilerplate header
If the upstream file has a long license text, paste the entire block without modifications.

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

# Your code begins here...
```

---

## Recommended Decision Rule

When deciding what header to use, follow this simple logic:
1. **Is it original?** If yes, use the short SPDX header (Case 1).
2. **Is it modified from a third party?** Use the Universal Template (Case 2): add our 3-line LoongForge header + origin statement, then copy-paste the original author's header verbatim below it.