# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Tokenizer-backed DeepSeek-V4.1 Engram hashing and gated residual memory."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.distributed as dist
from megatron.core.tensor_parallel.layers import set_tensor_model_parallel_attributes
from megatron.core.tensor_parallel.mappings import reduce_from_tensor_model_parallel_region
from torch import nn

from .deepseek_v41_linear import build_deepseek_v41_linear


def _linear_output(value):
    return value[0] if isinstance(value, tuple) else value


def _next_prime(start: int, used: set[int]) -> int:
    candidate = start + 1
    while True:
        if (
            candidate >= 2
            and (
                candidate == 2 or (candidate % 2 and all(candidate % n for n in range(3, math.isqrt(candidate) + 1, 2)))
            )
            and candidate not in used
        ):
            return candidate
        candidate += 1


def _compressed_token_map(tokenizer) -> tuple[tuple[int, ...], int]:
    try:
        from tokenizers import Regex, normalizers
    except ImportError as exc:
        raise RuntimeError("DeepSeek-V4.1 Engram requires the tokenizers package") from exc

    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    backend = tokenizer.backend_tokenizer
    key_to_id: dict[str, int] = {}
    token_map = []
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        if key not in key_to_id:
            key_to_id[key] = len(key_to_id)
        token_map.append(key_to_id[key])
    return tuple(token_map), len(key_to_id)


class DeepseekV41NgramHash(nn.Module):
    """Construct the released prime-bucket hash rows for all Engram layers."""

    def __init__(self, config, tokenizer):
        super().__init__()
        self.max_ngram_size = config.engram_max_ngram_size
        self.n_heads = config.engram_n_heads
        self.layer_ids = tuple(config.engram_layer_ids)
        if not self.layer_ids:
            raise ValueError("DeepSeek-V4.1 Engram hash requires at least one layer")
        token_map, compressed_vocab = _compressed_token_map(tokenizer)
        if compressed_vocab != config.engram_compressed_vocab_size:
            raise ValueError(
                "DeepSeek-V4.1 compressed tokenizer vocabulary mismatch: "
                f"got {compressed_vocab}, expected {config.engram_compressed_vocab_size}"
            )
        self.pad_id = token_map[config.engram_pad_token_id]
        primes, offsets, multipliers = [], [], []
        used: set[int] = set()
        multiplier_bound = max(1, (np.iinfo(np.int64).max // compressed_vocab) // 2)
        for layer_id, rows in zip(self.layer_ids, config.engram_num_embeddings):
            layer_primes = []
            for _ in range(self.max_ngram_size - 1):
                current = config.engram_vocab_size - 1
                for _ in range(self.n_heads):
                    current = _next_prime(current, used)
                    used.add(current)
                    layer_primes.append(current)
            if sum(layer_primes) > rows:
                raise ValueError(f"Engram layer {layer_id} needs {sum(layer_primes)} rows, configured {rows}")
            primes.append(tuple(layer_primes))
            offsets.append(tuple(int(v) for v in np.cumsum([0, *layer_primes[:-1]])))
            rng = np.random.default_rng(10007 * layer_id)
            values = rng.integers(0, multiplier_bound, size=(self.max_ngram_size,), dtype=np.int64)
            multipliers.append(tuple(int(v) for v in values * 2 + 1))
        self._token_map = token_map
        self._primes = tuple(primes)
        self._offsets = tuple(offsets)
        self._multipliers = tuple(multipliers)
        self.register_buffer("token_map", torch.tensor(token_map, dtype=torch.long), persistent=False)
        self.register_buffer("primes", torch.tensor(primes, dtype=torch.long), persistent=False)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long), persistent=False)
        self.register_buffer("multipliers", torch.tensor(multipliers, dtype=torch.long), persistent=False)
        self.register_load_state_dict_post_hook(self._restore_hash_buffers)

    def _restore_hash_buffers(self, module, incompatible_keys) -> None:
        # Nonpersistent buffers are empty after meta-device materialization.
        self.init_weights()

    @torch.no_grad()
    def init_weights(self) -> None:
        for name, values in (
            ("token_map", self._token_map),
            ("primes", self._primes),
            ("offsets", self._offsets),
            ("multipliers", self._multipliers),
        ):
            self.get_buffer(name).copy_(torch.tensor(values, dtype=torch.long, device=self.token_map.device))

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        token_mask: torch.Tensor | None = None,
        sequence_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Engram input_ids must be int32/int64 [batch, sequence]")
        if input_ids.numel() and bool(((input_ids < 0) | (input_ids >= self.token_map.numel())).any()):
            raise ValueError("Engram input_ids contains an out-of-vocabulary token")
        batch, sequence = input_ids.shape
        compressed = self.token_map[input_ids.long()]
        if token_mask is not None:
            if token_mask.shape != input_ids.shape:
                raise ValueError("Engram token_mask must match input_ids")
            compressed = torch.where(token_mask.bool(), compressed, -1)
        positions = torch.arange(sequence, device=input_ids.device).expand(batch, sequence)
        if sequence_ids is not None and sequence_ids.shape != input_ids.shape:
            raise ValueError("Engram sequence_ids must match input_ids")
        blocked = torch.zeros_like(compressed, dtype=torch.bool)
        tokens = []
        for shift in range(self.max_ngram_size):
            source_positions = positions - shift
            source = compressed.gather(1, source_positions.clamp_min(0))
            blocked = blocked | (source_positions < 0) | (source == -1)
            if sequence_ids is not None:
                source_ids = sequence_ids.gather(1, source_positions.clamp_min(0))
                blocked = blocked | (source_ids != sequence_ids) | (sequence_ids == 0)
            tokens.append(torch.where(blocked, self.pad_id, source))
        values = torch.stack(tokens, -1).unsqueeze(2) * self.multipliers
        rolling = values[..., 0]
        hashes = []
        for shift in range(1, self.max_ngram_size):
            rolling = torch.bitwise_xor(rolling, values[..., shift])
            start = (shift - 1) * self.n_heads
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, start : start + self.n_heads])
        return torch.cat(hashes, -1) + self.offsets


class DeepseekV41ParallelEmbedding(nn.Module):
    """Row-sharded trainable Engram table with an all-reduce lookup contract.

    The released table is stored as FP8 plus row-wise E8M0 scales.  The model
    keeps a BF16/FP16 parameter for continuation training and decodes the
    released representation while loading the checkpoint.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        process_group=None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.process_group = process_group
        self.compute_dtype = dtype
        self.world_size = (
            dist.get_world_size(process_group)
            if dist.is_available() and dist.is_initialized() and process_group is not None
            else 1
        )
        self.rank = dist.get_rank(process_group) if self.world_size > 1 else 0
        self.local_rows = (num_embeddings + self.world_size - 1) // self.world_size
        self.start = self.rank * self.local_rows
        self.end = self.start + self.local_rows
        self.weight = nn.Parameter(torch.empty(self.local_rows, embedding_dim, dtype=dtype))
        set_tensor_model_parallel_attributes(self.weight, True, 0, 1)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        # Initialize in place so the large row-sharded table never creates a
        # second FP32 copy during model construction.
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def _localize_rows(self, value: torch.Tensor, fill_value: float = 0.0) -> torch.Tensor:
        if value.ndim != 2:
            raise ValueError("DeepSeek-V4.1 Engram tensors must be two-dimensional")
        if value.shape[0] == self.local_rows:
            return value
        if value.shape[0] < self.start:
            raise ValueError(
                f"Engram checkpoint has {value.shape[0]} rows, cannot load rank range " f"[{self.start}, {self.end})"
            )
        stop = min(self.end, value.shape[0])
        local = torch.full(
            (self.local_rows, value.shape[1]),
            fill_value,
            dtype=value.dtype,
            device=value.device,
        )
        local[: stop - self.start].copy_(value[self.start : stop])
        return local

    def _decode_checkpoint_weight(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor | None,
    ) -> torch.Tensor:
        if weight.dtype != torch.float8_e4m3fn:
            return self._localize_rows(weight)
        if scale is None:
            raise ValueError("FP8 Engram checkpoint weight is missing its scale tensor")
        weight = self._localize_rows(weight)
        scale = self._localize_rows(scale, fill_value=1.0)
        if weight.shape[1] % 32:
            raise ValueError("DeepSeek-V4.1 Engram embedding width must be divisible by 32")
        decoded = torch.empty(
            weight.shape,
            dtype=self.weight.dtype,
            device=weight.device,
        )
        # Bound the temporary to keep the two very large released tables
        # loadable without creating another full-size FP32 copy.
        rows_per_chunk = max(1, (4 * 1024 * 1024) // weight.shape[1])
        for start in range(0, weight.shape[0], rows_per_chunk):
            stop = min(weight.shape[0], start + rows_per_chunk)
            values = weight[start:stop].float().unflatten(-1, (-1, 32))
            values = values * scale[start:stop].float().unsqueeze(-1)
            decoded[start:stop].copy_(values.flatten(-2).to(self.weight.dtype))
        return decoded

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        key = prefix + "weight"
        value = state_dict.get(key)
        scale_key = prefix + "scale"
        scale = state_dict.pop(scale_key, None)
        if value is not None:
            state_dict[key] = self._decode_checkpoint_weight(value, scale)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        mask = (indices < self.start) | (indices >= self.end)
        local_indices = (indices - self.start).masked_fill(mask, 0)
        values = self.weight[local_indices].float()
        values = values.masked_fill(mask.unsqueeze(-1), 0)
        if self.world_size > 1:
            values = reduce_from_tensor_model_parallel_region(values, group=self.process_group)
        return values.to(self.compute_dtype)


class DeepseekV41Engram(nn.Module):
    """Lookup hash rows and inject the V4.1 signed square-root gated residual."""

    def __init__(self, config, layer_id: int, process_group=None):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.streams = config.hc_mult
        self.hash_heads = (config.engram_max_ngram_size - 1) * config.engram_n_heads
        self.head_dim = config.engram_head_dim
        table_index = list(config.engram_layer_ids).index(layer_id)
        self.hash_index = table_index
        self.num_embeddings = config.engram_num_embeddings[table_index]
        self.embed = DeepseekV41ParallelEmbedding(
            self.num_embeddings,
            self.head_dim,
            process_group=process_group,
            dtype=getattr(config, "params_dtype", torch.bfloat16),
        )
        self.wkv = build_deepseek_v41_linear(
            config,
            self.hash_heads * self.head_dim,
            self.streams * self.hidden_size + self.hidden_size,
            mode="duplicated",
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
        )
        self.q_weight = nn.Parameter(torch.ones(self.streams, self.hidden_size, dtype=config.params_dtype))
        self.k_weight = nn.Parameter(torch.ones(self.streams, self.hidden_size, dtype=config.params_dtype))
        self.eps = config.layernorm_epsilon

    def forward(
        self,
        hidden_states: torch.Tensor,
        hash_ids: torch.Tensor,
        *,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_states.ndim != 4:
            raise ValueError("Engram hidden states must be [batch, sequence, streams, hidden]")
        if hash_ids.shape != (
            hidden_states.shape[0],
            hidden_states.shape[1],
            self.hash_heads,
        ):
            raise ValueError("Engram hash IDs do not match hidden states")
        values = self.embed(hash_ids.long()).flatten(-2)
        key, value = _linear_output(self.wkv(values)).split([self.streams * self.hidden_size, self.hidden_size], dim=-1)
        streams = hidden_states.float()
        key = key.float().view(*key.shape[:2], self.streams, self.hidden_size)
        weights = self.q_weight.float() * self.k_weight.float()
        rstd = torch.rsqrt(streams.square().mean(-1) + self.eps)
        rstd = rstd * torch.rsqrt(key.square().mean(-1) + self.eps)
        dot = (streams * weights * key).sum(-1) * rstd * self.hidden_size**-0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.bool().unsqueeze(-1), 0)
        streams = streams + gate.unsqueeze(-1) * value.float().unsqueeze(-2)
        return streams.to(hidden_states.dtype)
