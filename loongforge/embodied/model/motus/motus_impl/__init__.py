# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from https://github.com/thu-ml/Motus under the Apache-2.0 License.
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

"""Vendored Motus model implementation (transplanted verbatim from the source repo).

Contents
--------
- ``motus.py``          : ``Motus`` / ``MotusConfig`` (three-modal MoT)
- ``wan_model.py``      : WAN video backbone + Wan2.2 VAE wrapper
- ``action_expert.py``  : Action Expert (DiT + cross-attention)
- ``und_expert.py``     : Understanding Expert
- ``wan/``              : the WAN package (attention/model/vae/schedulers), unmodified
- ``motus_utils.py``    : the subset of the source ``utils.common`` needed by the model

Only import-path lines were edited (absolute ``wan.*`` / ``utils.common`` -> package
relative; source ``sys.path``/``bak`` hacks removed). Numerics are unchanged: the
real-valued rope, VAE first-frame reuse, and ``num_warmup_iters=20`` are all preserved.

Heavy dependencies (transformers, the WAN backbone) are imported only when
``motus.py`` is imported, which is done lazily from ``modeling_motus.py``.
"""
