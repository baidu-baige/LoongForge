import math

import torch
import torch.nn.functional as F  # noqa: N812
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from giga_models.utils.action_horizon import resolve_flow_action_steps
from .paligemma2_with_expert import PaliGemma2WithExpertModel

# ── Fine-grained NVTX / torch.profiler range markers (opt-in, gated) ──
# Zero overhead and zero behavior change when the switch is off.
try:
    from loongforge.embodied.profiling_nvtx import nvtx_range, nvtx_push, nvtx_pop, set_enabled
except Exception:  # pragma: no cover - profiling is optional
    import contextlib as _contextlib

    def set_enabled(_flag):  # type: ignore
        return None

    def nvtx_push(_msg):  # type: ignore
        return None

    def nvtx_pop():  # type: ignore
        return None

    @_contextlib.contextmanager
    def nvtx_range(_msg):  # type: ignore
        yield


class _GradientWeight(torch.autograd.Function):
    """Identity in forward, scalar gradient weight in backward."""

    @staticmethod
    def forward(ctx, tensor: Tensor, weight: float) -> Tensor:
        ctx.weight = weight
        return tensor

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        return grad_output * ctx.weight, None


def _apply_gradient_weight(tensor: Tensor, weight: float) -> Tensor:
    if weight == 0.0:
        return tensor.detach()
    if weight == 1.0:
        return tensor
    return _GradientWeight.apply(tensor, weight)


def _weight_kv_cache_gradients(past_key_values: dict, weight: float) -> None:
    """Weight gradients crossing from an action suffix into the VLM prefix."""
    for layer_cache in past_key_values.values():
        layer_cache['key_states'] = _apply_gradient_weight(layer_cache['key_states'], weight)
        layer_cache['value_states'] = _apply_gradient_weight(layer_cache['value_states'], weight)


def _repeat_kv_cache(past_key_values: dict, repeats: int, gradient_weight: float) -> dict:
    """Expand prefix KV while preserving the configured action-to-VLM
    gradient."""
    repeated_cache = {}
    for layer_idx, layer_cache in past_key_values.items():
        repeated_layer_cache = dict(layer_cache)
        for state_name in ('key_states', 'value_states'):
            state = _apply_gradient_weight(layer_cache[state_name], gradient_weight)
            repeated_layer_cache[state_name] = state.repeat_interleave(repeats, dim=0)
        repeated_cache[layer_idx] = repeated_layer_cache
    return repeated_cache


def _compact_language_logits(
    model: nn.Module,
    hidden_states: Tensor,
    lang_loss_masks: Tensor | None,
) -> dict[str, Tensor]:
    if lang_loss_masks is None:
        return {'lang_logits': model.language_out_proj(hidden_states)}

    if hidden_states.shape[:2] != lang_loss_masks.shape:
        raise ValueError(
            f'hidden_states prefix shape {tuple(hidden_states.shape[:2])} does not match ' f'lang_loss_masks shape {tuple(lang_loss_masks.shape)}'
        )

    loss_positions = lang_loss_masks[:, 1:].to(device=hidden_states.device, dtype=torch.bool)
    hidden_for_loss = hidden_states[:, :-1, :][loss_positions]
    return {
        'lang_logits': model.language_out_proj(hidden_for_loss),
        'lang_logits_mask': loss_positions,
    }


def _chunked_language_token_loss(
    model: nn.Module,
    hidden_states: Tensor,
    lang_tokens: Tensor,
    lang_loss_masks: Tensor,
    chunk_size: int,
    use_checkpoint: bool,
) -> dict[str, Tensor]:
    if hidden_states.shape[:2] != lang_loss_masks.shape:
        raise ValueError(
            f'hidden_states prefix shape {tuple(hidden_states.shape[:2])} does not match ' f'lang_loss_masks shape {tuple(lang_loss_masks.shape)}'
        )
    if hidden_states.shape[:2] != lang_tokens.shape:
        raise ValueError(
            f'hidden_states prefix shape {tuple(hidden_states.shape[:2])} does not match ' f'lang_tokens shape {tuple(lang_tokens.shape)}'
        )

    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError(f'chunk_size must be positive, got {chunk_size}')

    loss_positions = lang_loss_masks[:, 1:].to(device=hidden_states.device, dtype=torch.bool)
    hidden_for_loss = hidden_states[:, :-1, :][loss_positions]
    targets = lang_tokens[:, 1:].to(device=hidden_states.device)[loss_positions].long()

    if hidden_for_loss.shape[0] == 0:
        token_loss = hidden_states[:, :-1, 0].float() * 0.0
        return {'lang_token_loss': token_loss}

    def _ce_for_hidden(hidden_chunk: Tensor, target_chunk: Tensor) -> Tensor:
        logits = model.language_out_proj(hidden_chunk)
        return F.cross_entropy(logits, target_chunk, reduction='none')

    losses = []
    for start in range(0, hidden_for_loss.shape[0], chunk_size):
        end = min(start + chunk_size, hidden_for_loss.shape[0])
        hidden_chunk = hidden_for_loss[start:end]
        target_chunk = targets[start:end]
        if use_checkpoint and hidden_chunk.requires_grad:
            losses.append(checkpoint(_ce_for_hidden, hidden_chunk, target_chunk, use_reentrant=False))
        else:
            losses.append(_ce_for_hidden(hidden_chunk, target_chunk))

    compact_loss = torch.cat(losses, dim=0)
    token_loss = compact_loss.new_zeros(loss_positions.shape)
    token_loss[loss_positions] = compact_loss
    return {'lang_token_loss': token_loss}


def _mask_action_suffix_target_attention(
    full_att_2d: Tensor,
    suffix_len: int,
    fast_action_indicator: Tensor | None,
    subtask_indicator: Tensor | None,
) -> Tensor:
    """Keep flow-action suffix independent from autoregressive target
    tokens."""
    if suffix_len <= 0:
        return full_att_2d

    batch_size = full_att_2d.shape[0]
    start = full_att_2d.shape[1] - suffix_len
    full_fast_indicator = None

    if fast_action_indicator is not None:
        suffix_fast_indicator = fast_action_indicator.new_zeros(batch_size, suffix_len)
        full_fast_indicator = torch.cat([fast_action_indicator, suffix_fast_indicator], dim=1)
        # Continuous flow action tokens must not condition on FAST action targets.
        full_att_2d[:, start:, :] &= ~full_fast_indicator[:, None, :]

    if subtask_indicator is not None:
        suffix_subtask_indicator = subtask_indicator.new_zeros(batch_size, suffix_len)
        full_subtask_indicator = torch.cat([subtask_indicator, suffix_subtask_indicator], dim=1)
        # Continuous flow action tokens must not condition on predicted subtask targets.
        full_att_2d[:, start:, :] &= ~full_subtask_indicator[:, None, :]

        if full_fast_indicator is not None:
            # FAST action target tokens also stay independent from predicted subtask targets.
            full_att_2d &= ~(full_fast_indicator[:, :, None] & full_subtask_indicator[:, None, :])

    return full_att_2d


class GigaBrain07Policy(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        max_state_dim: int = 32,
        max_action_dim: int = 32,
        proj_width: int = 1024,
        vlm_type: str = 'paligemma2',
        vlm_hidden_size: int = 2304,
        n_action_steps: int = 50,
        flow_action_horizon: int | None = None,
        num_steps: int = 10,
        use_cache: bool = True,
        observation_memory_size: int = 1,
        state_input_mode: str = 'prompt',
        agent_pos_config: dict | None = None,
        state_hidden_size: int = 2048,
        proj_with_mask: bool = True,
        propri_token_id: int | None = None,
        vision_in_channels: int = 3,
        temporal_attention_interval: int = 4,
        detach_history_frames: bool = True,
        enable_knowledge_insulation: bool = False,
        enable_next_token_prediction: bool = True,
        compact_language_logits: bool = True,
        language_loss_chunk_size: int = 0,
        language_loss_checkpoint: bool = True,
        enable_learnable_traj_token: bool = False,
        num_traj_tokens: int = 10,
        max_traj_dim: int = 4,
        traj_hidden_dim: int = 256,
        num_embodiments: int = 1,
        has_action_expert: bool = False,
        fast_token_vocab_mode: str | None = None,
        fast_token_tail_skip_tokens: int = 128,
        fast_token_tail_vocab_size: int | None = None,
        fast_vocab_size: int = 2048,
        # PaliGemma2 only: gradient weight at the action-expert -> VLM prefix
        # boundary. None preserves the legacy boolean behavior (0.0 with KI,
        # 1.0 without KI). An explicit value takes precedence over the boolean.
        action_loss_vlm_gradient_weight: float | None = None,
        enable_nvtx_profiling: bool = False,
        **kwargs,
    ):
        super().__init__()
        if vlm_type != 'paligemma2':
            raise ValueError(f"GigaBrain-0.7 requires vlm_type='paligemma2', got {vlm_type!r}")
        self.has_action_expert = has_action_expert
        # Wire the process-wide NVTX/profiler switch at build time. env
        # LOONGFORGE_NVTX=1 force-on still wins inside set_enabled().
        self.enable_nvtx_profiling = bool(enable_nvtx_profiling)
        set_enabled(self.enable_nvtx_profiling)

        # Store the parameters
        self.max_state_dim = max_state_dim
        if int(max_action_dim) != 32:
            raise ValueError(f'GigaBrain action heads require max_action_dim=32, got {max_action_dim}')
        self.max_action_dim = 32
        self._action_denoise_mask: Tensor | None = None
        self.proj_width = proj_width
        _, self.n_action_steps, self.flow_action_horizon = resolve_flow_action_steps(
            n_action_steps,
            flow_action_horizon,
        )
        self.num_steps = num_steps
        self.use_cache = use_cache
        if observation_memory_size < 1:
            raise ValueError(f'observation_memory_size must be positive, got {observation_memory_size}')
        self.observation_memory_size = observation_memory_size
        if state_input_mode not in ('prompt', 'proprio_memory', 'proprio_anchor'):
            raise ValueError("state_input_mode must be 'prompt', 'proprio_memory', or " f"'proprio_anchor', got {state_input_mode!r}")
        self.state_input_mode = state_input_mode
        self.agent_pos_config = dict(agent_pos_config or {})
        self.propri_dim = sum(int(dim) for dim in self.agent_pos_config.values())
        self.state_hidden_size = int(state_hidden_size)
        self.proj_with_mask = bool(proj_with_mask)
        self.propri_token_id = None if propri_token_id is None else int(propri_token_id)
        if self.state_input_mode == 'proprio_anchor':
            if vlm_type != 'paligemma2':
                raise ValueError("state_input_mode='proprio_anchor' is currently supported only for " f"vlm_type='paligemma2', got {vlm_type!r}")
            # Observation memory controls visual history; the anchor remains one token.
            if self.propri_dim <= 0:
                raise ValueError("state_input_mode='proprio_anchor' requires a non-empty agent_pos_config")
            if self.state_hidden_size <= 0 or self.state_hidden_size > int(vlm_hidden_size):
                raise ValueError(
                    'state_hidden_size must be positive and no larger than vlm_hidden_size, '
                    f'got state_hidden_size={self.state_hidden_size}, vlm_hidden_size={vlm_hidden_size}'
                )
            if self.propri_token_id is None:
                raise ValueError("state_input_mode='proprio_anchor' requires propri_token_id from the tokenizer")
        self.vision_in_channels = vision_in_channels
        self.temporal_attention_interval = temporal_attention_interval

        self.enable_knowledge_insulation = enable_knowledge_insulation
        if action_loss_vlm_gradient_weight is not None and vlm_type != 'paligemma2':
            raise ValueError('action_loss_vlm_gradient_weight is currently supported only for ' f"vlm_type='paligemma2', got {vlm_type!r}")
        if action_loss_vlm_gradient_weight is None:
            action_loss_vlm_gradient_weight = 0.0 if enable_knowledge_insulation else 1.0
        action_loss_vlm_gradient_weight = float(action_loss_vlm_gradient_weight)
        if not math.isfinite(action_loss_vlm_gradient_weight) or action_loss_vlm_gradient_weight < 0.0:
            raise ValueError(
                'action_loss_vlm_gradient_weight must be finite and greater than or equal to 0, ' f'got {action_loss_vlm_gradient_weight}'
            )
        self.action_loss_vlm_gradient_weight = action_loss_vlm_gradient_weight
        # @register_to_config already stored the caller-provided value. Keep None in
        # the config so a later KI override can resolve a new legacy default.
        self.vlm_type = vlm_type
        self.vlm_hidden_size = vlm_hidden_size
        if fast_token_vocab_mode is None:
            fast_token_vocab_mode = 'tail'
        self.fast_token_vocab_mode = fast_token_vocab_mode
        if self.fast_token_vocab_mode not in ('expanded', 'tail'):
            raise ValueError(f"fast_token_vocab_mode must be 'expanded' or 'tail', got {self.fast_token_vocab_mode!r}")
        self.fast_token_tail_skip_tokens = int(fast_token_tail_skip_tokens)
        self.fast_token_tail_vocab_size = None if fast_token_tail_vocab_size is None else int(fast_token_tail_vocab_size)
        self.fast_vocab_size = int(fast_vocab_size)
        self.register_to_config(fast_token_vocab_mode=self.fast_token_vocab_mode)

        self.paligemma_with_expert = PaliGemma2WithExpertModel(
            vision_in_channels=vision_in_channels,
            enable_next_token_prediction=enable_next_token_prediction,
            temporal_attention_interval=temporal_attention_interval,
            detach_history_frames=detach_history_frames,
        )

        # Projections are float32
        if self.state_input_mode == 'proprio_memory':
            self.proprio_state_proj = EmbodimentSpecificLinear(
                self.max_state_dim,
                self.vlm_hidden_size,
                num_categories=num_embodiments,
                dtype=torch.float32,
            )
        else:
            self.proprio_state_proj = None
        if self.state_input_mode == 'proprio_anchor':
            propri_input_dim = self.propri_dim * (2 if self.proj_with_mask else 1)
            self.propri_proj = nn.Linear(
                propri_input_dim,
                self.state_hidden_size,
                bias=False,
                dtype=torch.float32,
            )
            nn.init.normal_(self.propri_proj.weight, mean=0.0, std=0.02)
            num_embeddings = self.paligemma_with_expert.embed_tokens.num_embeddings
            if not 0 <= self.propri_token_id < num_embeddings:
                raise ValueError(f'propri_token_id={self.propri_token_id} is outside the PaliGemma2 ' f'embedding table with {num_embeddings} rows')
        else:
            self.propri_proj = None
        self.action_in_proj = EmbodimentSpecificLinear(self.max_action_dim, self.proj_width, num_categories=num_embodiments, dtype=torch.float32)
        self.action_out_proj = EmbodimentSpecificLinear(self.proj_width, self.max_action_dim, num_categories=num_embodiments, dtype=torch.float32)
        self.time_mlp_in = nn.Linear(self.proj_width, self.proj_width, dtype=torch.float32)
        self.time_mlp_out = nn.Linear(self.proj_width, self.proj_width, dtype=torch.float32)

        self.enable_next_token_prediction = enable_next_token_prediction
        self.compact_language_logits = compact_language_logits
        self.language_loss_chunk_size = int(language_loss_chunk_size)
        self.language_loss_checkpoint = bool(language_loss_checkpoint)
        self.enable_learnable_traj_token = enable_learnable_traj_token
        if self.enable_learnable_traj_token:
            self.num_traj_tokens = num_traj_tokens
            self.max_traj_dim = max_traj_dim
            self.traj_hidden_dim = traj_hidden_dim
            self.traj_token = nn.Parameter(torch.randn(num_traj_tokens, self.vlm_hidden_size), requires_grad=True)
            self.traj_decoder = nn.GRU(input_size=self.vlm_hidden_size, hidden_size=self.traj_hidden_dim, batch_first=True)
            self.traj_out_proj = nn.Linear(self.traj_hidden_dim, self.max_traj_dim, dtype=torch.float32)

        # ───────────────────────────── eval / inference entry ─────────────────────────────
        # Runtime configuration consumed by ``predict_action`` (the shared eval
        # entry point). These are plain attributes, deliberately NOT
        # @register_to_config fields, mirroring how ``XVLAPolicy`` keeps
        # ``_processor_path`` / ``_num_image_views`` off the serialized config.
        # An eval factory wires them after build via ``configure_eval``; the
        # defaults keep a freshly built model importable and introspectable
        # without any eval wiring.
        self._processor_path: str = ''
        self._eval_prompt_cfg: dict | None = None
        self._eval_resize_imgs_with_padding: tuple[int, int] | None = None
        self._eval_present_img_keys: list[str] | None = None
        self._eval_max_length: int = 256
        self._eval_discrete_state_input: bool = True
        self._image_transform = None
        self._prompt_tokenizer_transform = None

    def _embed_proprio_state_tokens(
        self,
        state_memory: Tensor | None,
        state_memory_masks: Tensor | None,
        emb_ids: Tensor | None,
        dtype: torch.dtype,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        if self.state_input_mode != 'proprio_memory':
            return None, None, None
        if self.proprio_state_proj is None:
            raise RuntimeError('proprio_state_proj is not initialized for proprio_memory mode')
        if state_memory is None:
            raise ValueError("state_input_mode='proprio_memory' requires state_memory")
        if emb_ids is None:
            raise ValueError("state_input_mode='proprio_memory' requires emb_ids for embodiment-specific projection")

        if state_memory.ndim == 2:
            state_memory = state_memory[:, None, :]
        if state_memory.ndim != 3:
            raise ValueError(f'state_memory must have shape [B,K,D], got {tuple(state_memory.shape)}')
        batch_size, memory_size, state_dim = state_memory.shape
        if memory_size != self.observation_memory_size:
            raise ValueError(f'state_memory has K={memory_size}, but observation_memory_size={self.observation_memory_size}')

        if state_dim < self.max_state_dim:
            pad_width = self.max_state_dim - state_dim
            state_memory = F.pad(state_memory, (0, pad_width), value=0.0)
        elif state_dim > self.max_state_dim:
            state_memory = state_memory[..., : self.max_state_dim]

        if state_memory_masks is None:
            state_memory_masks = torch.ones(
                batch_size,
                memory_size,
                dtype=torch.bool,
                device=state_memory.device,
            )
        else:
            state_memory_masks = state_memory_masks.to(device=state_memory.device, dtype=torch.bool)
            if state_memory_masks.shape != (batch_size, memory_size):
                raise ValueError(
                    f'state_memory_masks shape {tuple(state_memory_masks.shape)} does not match ' f'state_memory shape {(batch_size, memory_size)}'
                )

        state_memory = state_memory.to(dtype=self.proprio_state_proj.weight.dtype)
        state_embs = self.proprio_state_proj(state_memory, emb_ids=emb_ids).to(dtype=dtype)
        state_att_masks = torch.zeros(
            batch_size,
            memory_size,
            dtype=torch.bool,
            device=state_memory.device,
        )
        return state_embs, state_memory_masks, state_att_masks

    def _project_proprioception_anchor(
        self,
        proprioception: Tensor | None,
        agent_pos_mask: Tensor | None,
        *,
        dtype: torch.dtype,
    ) -> Tensor | None:
        if self.state_input_mode != 'proprio_anchor':
            return None
        if self.propri_proj is None:
            raise RuntimeError('propri_proj is not initialized for proprio_anchor mode')
        if proprioception is None:
            raise ValueError("state_input_mode='proprio_anchor' requires proprioception")

        if proprioception.ndim == 2:
            proprioception = proprioception[:, None, :]
        expected_shape = (proprioception.shape[0], 1, self.propri_dim)
        if tuple(proprioception.shape) != expected_shape:
            raise ValueError(f'proprioception must have shape [B,1,{self.propri_dim}], ' f'got {tuple(proprioception.shape)}')

        projector_input = proprioception.to(
            device=self.propri_proj.weight.device,
            dtype=self.propri_proj.weight.dtype,
        )
        if self.proj_with_mask:
            if agent_pos_mask is None:
                raise ValueError('proj_with_mask=True requires agent_pos_mask')
            if agent_pos_mask.ndim == 2:
                agent_pos_mask = agent_pos_mask[:, None, :]
            if tuple(agent_pos_mask.shape) != expected_shape:
                raise ValueError(f'agent_pos_mask must have shape [B,1,{self.propri_dim}], ' f'got {tuple(agent_pos_mask.shape)}')
            projector_input = torch.cat(
                [
                    projector_input,
                    agent_pos_mask.to(
                        device=projector_input.device,
                        dtype=projector_input.dtype,
                    ),
                ],
                dim=-1,
            )

        state_hidden = self.propri_proj(projector_input)
        if self.state_hidden_size < self.vlm_hidden_size:
            state_hidden = F.pad(
                state_hidden,
                (0, self.vlm_hidden_size - self.state_hidden_size),
                value=0.0,
            )
        return state_hidden.to(dtype=dtype)

    def _scatter_proprioception_anchor(
        self,
        lang_tokens: Tensor,
        lang_embs: Tensor,
        proprioception: Tensor | None,
        agent_pos_mask: Tensor | None,
        proprioception_present: Tensor | None = None,
    ) -> Tensor:
        if self.state_input_mode != 'proprio_anchor':
            return lang_embs

        if lang_tokens.ndim != 2:
            raise ValueError(f'lang_tokens must have shape [B,L], got {tuple(lang_tokens.shape)}')
        if lang_embs.ndim != 3 or tuple(lang_embs.shape[:2]) != tuple(lang_tokens.shape):
            raise ValueError(
                'lang_embs must have shape [B,L,H] matching lang_tokens; '
                f'got lang_tokens={tuple(lang_tokens.shape)}, lang_embs={tuple(lang_embs.shape)}'
            )

        positions = lang_tokens == self.propri_token_id
        counts = positions.sum(dim=1)
        if proprioception_present is None:
            expected_counts = torch.ones_like(counts)
        else:
            expected_counts = proprioception_present.to(
                device=counts.device,
                dtype=torch.bool,
            )
            if expected_counts.ndim == 0 and counts.shape[0] == 1:
                expected_counts = expected_counts[None]
            if tuple(expected_counts.shape) != tuple(counts.shape):
                raise ValueError(
                    'proprioception_present must have shape [B], ' f'got {tuple(expected_counts.shape)} for batch size {counts.shape[0]}'
                )
            expected_counts = expected_counts.to(dtype=counts.dtype)
        if not torch.equal(counts, expected_counts):
            raise ValueError(
                'Unexpected <|propri|> count per sample: '
                f'expected {expected_counts.detach().cpu().tolist()}, '
                f'got {counts.detach().cpu().tolist()}'
            )
        if not torch.any(expected_counts):
            return lang_embs
        state_embed = self._project_proprioception_anchor(
            proprioception,
            agent_pos_mask,
            dtype=lang_embs.dtype,
        )
        if state_embed is None:
            raise RuntimeError('proprio_anchor projection unexpectedly returned None')
        expected_state_shape = (
            lang_tokens.shape[0],
            1,
            lang_embs.shape[-1],
        )
        if tuple(state_embed.shape) != expected_state_shape:
            raise ValueError(
                'Projected proprioception must match the language batch and hidden size; '
                f'expected {expected_state_shape}, got {tuple(state_embed.shape)}'
            )
        return torch.where(positions.unsqueeze(-1), state_embed, lang_embs)

    def _prepend_proprio_prefix_tokens(
        self,
        prefix_embs: Tensor,
        prefix_pad_masks: Tensor,
        prefix_att_masks: Tensor,
        state_memory: Tensor | None,
        state_memory_masks: Tensor | None,
        emb_ids: Tensor | None,
        fast_action_indicator: Tensor | None = None,
        subtask_indicator: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor | None, Tensor | None]:
        state_embs, state_pad_masks, state_att_masks = self._embed_proprio_state_tokens(
            state_memory,
            state_memory_masks,
            emb_ids,
            dtype=prefix_embs.dtype,
        )
        if state_embs is None:
            return prefix_embs, prefix_pad_masks, prefix_att_masks, fast_action_indicator, subtask_indicator, None

        state_att_masks = state_att_masks.to(dtype=prefix_att_masks.dtype)
        prefix_embs = torch.cat([state_embs, prefix_embs], dim=1)
        prefix_pad_masks = torch.cat([state_pad_masks, prefix_pad_masks], dim=1)
        prefix_att_masks = torch.cat([state_att_masks, prefix_att_masks], dim=1)
        state_len = state_embs.shape[1]
        if fast_action_indicator is not None:
            fast_action_indicator = F.pad(fast_action_indicator, (state_len, 0), value=False)
        if subtask_indicator is not None:
            subtask_indicator = F.pad(subtask_indicator, (state_len, 0), value=False)
        return prefix_embs, prefix_pad_masks, prefix_att_masks, fast_action_indicator, subtask_indicator, state_pad_masks

    @staticmethod
    def _prepend_proprio_position_ids(position_ids: Tensor, state_pad_masks: Tensor | None) -> Tensor:
        if state_pad_masks is None:
            return position_ids

        state_pos = (torch.cumsum(state_pad_masks.long(), dim=1) - 1).clamp_min(0)
        state_offsets = state_pad_masks.long().sum(dim=1)
        if position_ids.ndim == 3:
            state_pos = state_pos.unsqueeze(0).expand(position_ids.shape[0], -1, -1)
            offset = state_offsets.view(1, -1, 1)
            return torch.cat([state_pos, position_ids + offset], dim=-1)

        offset = state_offsets.view(-1, 1)
        return torch.cat([state_pos, position_ids + offset], dim=1)

    @staticmethod
    def _last_valid_indices(pad_masks: Tensor) -> Tensor:
        positions = torch.arange(pad_masks.shape[1], device=pad_masks.device)
        positions = positions.view(1, -1).expand_as(pad_masks)
        return positions.masked_fill(~pad_masks.to(dtype=torch.bool), 0).max(dim=1).values

    def _attach_full_prefill_generation_state(
        self,
        state: dict,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        lang_att_masks: Tensor | None,
        image_grid_thw: Tensor | None,
        emb_ids: Tensor | None,
        state_memory: Tensor | None,
        state_memory_masks: Tensor | None,
    ) -> dict:
        """Store enough context to decode by recomputing the whole prefix.

        Some expert-backed backbones expose VLM-prefix prefill for flow inference but no token-by-token VLM decode API. For autoregressive FAST tokens
        with continuous proprio prefix tokens, recomputing the compact prefix is slower but preserves the exact observation prefix layout.
        """
        if lang_tokens.shape[0] != 1:
            raise NotImplementedError('full-prefix autoregressive fallback currently supports batch_size=1')

        valid_len = int(lang_masks[0].sum().item())
        state['full_prefill_decode'] = True
        state['images'] = images
        state['img_masks'] = img_masks
        state['lang_tokens'] = lang_tokens[:, :valid_len]
        state['lang_masks'] = lang_masks[:, :valid_len]
        state['lang_att_masks'] = None if lang_att_masks is None else lang_att_masks[:, :valid_len]
        state['image_grid_thw'] = image_grid_thw
        state['emb_ids'] = emb_ids
        state['state_memory'] = state_memory
        state['state_memory_masks'] = state_memory_masks
        return state

    @torch.no_grad()
    def _next_lang_logits_full_prefill(
        self,
        state: dict,
        input_token: Tensor,
    ) -> tuple[Tensor, dict]:
        if input_token.dtype != torch.long:
            input_token = input_token.to(torch.long)
        if input_token.dim() == 1:
            input_token = input_token.unsqueeze(1)

        lang_tokens = torch.cat([state['lang_tokens'], input_token], dim=1)
        new_mask = torch.ones(
            lang_tokens.shape[0],
            1,
            dtype=state['lang_masks'].dtype,
            device=state['lang_masks'].device,
        )
        lang_masks = torch.cat([state['lang_masks'], new_mask], dim=1)
        lang_att_masks = state['lang_att_masks']
        if lang_att_masks is not None:
            new_att = torch.ones(
                lang_att_masks.shape[0],
                1,
                dtype=lang_att_masks.dtype,
                device=lang_att_masks.device,
            )
            lang_att_masks = torch.cat([lang_att_masks, new_att], dim=1)

        return self.init_lang_generation(
            state['images'],
            state['img_masks'],
            lang_tokens,
            lang_masks,
            image_grid_thw=state.get('image_grid_thw'),
            lang_att_masks=lang_att_masks,
            emb_ids=state.get('emb_ids'),
            state_memory=state.get('state_memory'),
            state_memory_masks=state.get('state_memory_masks'),
        )

    def forward(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        x_t: Tensor,
        timestep: Tensor,
        emb_ids: Tensor,
        lang_att_masks: Tensor | None = None,
        fast_action_indicator: Tensor | None = None,
        subtask_indicator: Tensor | None = None,
        image_grid_thw: Tensor | None = None,
        lang_loss_masks: Tensor | None = None,
        state_memory: Tensor | None = None,
        state_memory_masks: Tensor | None = None,
        proprioception: Tensor | None = None,
        agent_pos_mask: Tensor | None = None,
        proprioception_present: Tensor | None = None,
        state: Tensor | None = None,
        action_emb_ids: Tensor | None = None,
        flow_matching_batch_mul: int = 1,
    ) -> dict[str, Tensor]:
        """Runs the full model forward pass.

        This method processes image and language inputs, generates key-value caches,
        and then uses them to denoise an action sequence `x_t` at a given `timestep`.
        It also computes language logits and, if enabled, trajectory predictions.

        Args:
            images: A list of image tensors.
            img_masks: A list of boolean masks for the images.
            lang_tokens: Language input token IDs.
            lang_masks: Boolean mask for the language tokens.
            x_t: The noisy action tensor at the current timestep.
            timestep: The current timestep value.
            emb_ids: Embodiment IDs for selecting embodiment-specific layers.
            lang_att_masks: Optional attention masks for language tokens.
            fast_action_indicator: Optional tensor indicating fast action token positions.
            subtask_indicator: Optional tensor indicating subtask token positions.
              When both are provided, FAST token rows are blocked from attending to
              subtask token columns so the two groups are independent of each other.
            image_grid_thw: Reserved compatibility argument.
            action_emb_ids: Embodiment IDs aligned with the possibly expanded
                action batch. Defaults to emb_ids for the standard K=1 path.
            flow_matching_batch_mul: Number of independent noisy action samples
                paired with each prefix.

        Returns:
            A dictionary containing:
            - 'v_t': The predicted velocity (denoised action).
            - 'lang_logits': Logits for the language model output.
            - 'traj_pred' (optional): The predicted trajectory if enabled.
        """
        if flow_matching_batch_mul < 1:
            raise ValueError(f'flow_matching_batch_mul must be positive, but got {flow_matching_batch_mul}')
        if flow_matching_batch_mul > 1 and self.vlm_type != 'paligemma2':
            raise NotImplementedError(
                'Multi-sample flow matching currently supports only the PaliGemma2 action-expert path, ' f'but got vlm_type={self.vlm_type!r}'
            )
        if action_emb_ids is None:
            action_emb_ids = emb_ids

        lang_length = lang_masks.shape[-1]

        nvtx_push('embed_prefix')
        prefix_embs, prefix_pad_masks, prefix_att_masks, fast_action_indicator, subtask_indicator, traj_token_start_idx = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            lang_att_masks,
            fast_action_indicator,
            subtask_indicator,
            proprioception=proprioception,
            agent_pos_mask=agent_pos_mask,
            proprioception_present=proprioception_present,
        )
        (
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            fast_action_indicator,
            subtask_indicator,
            state_pad_masks,
        ) = self._prepend_proprio_prefix_tokens(
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            state_memory,
            state_memory_masks,
            emb_ids,
            fast_action_indicator,
            subtask_indicator,
        )
        if state_pad_masks is not None and traj_token_start_idx is not None:
            traj_token_start_idx += state_pad_masks.shape[1]
        nvtx_pop()  # embed_prefix
        nvtx_push('embed_suffix')
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            x_t,
            timestep,
            emb_ids=action_emb_ids,
        )
        nvtx_pop()  # embed_suffix

        if flow_matching_batch_mul > 1:
            return self._forward_paligemma2_multi_flow_samples(
                prefix_embs=prefix_embs,
                prefix_pad_masks=prefix_pad_masks,
                prefix_att_masks=prefix_att_masks,
                suffix_embs=suffix_embs,
                suffix_pad_masks=suffix_pad_masks,
                suffix_att_masks=suffix_att_masks,
                adarms_cond=adarms_cond,
                action_emb_ids=action_emb_ids,
                fast_action_indicator=fast_action_indicator,
                subtask_indicator=subtask_indicator,
                lang_length=lang_length,
                traj_token_start_idx=traj_token_start_idx,
                batch_mul=flow_matching_batch_mul,
            )

        # Construct 2d attention mask and position ids
        batch_size, prefix_len = prefix_pad_masks.shape
        suffix_len = suffix_pad_masks.shape[1]

        pad_masks_full = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks_full = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        full_att_2d = make_att_2d_masks(pad_masks_full, att_masks_full)

        full_att_2d = _mask_action_suffix_target_attention(
            full_att_2d,
            suffix_len,
            fast_action_indicator,
            subtask_indicator,
        )

        prefix_position_ids = self._compute_1d_position_ids(prefix_pad_masks)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1, keepdim=True)
        suffix_position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        if fast_action_indicator is not None:
            suffix_position_ids = suffix_position_ids - fast_action_indicator.sum(dim=-1, keepdim=True)

        # VLM forward pass
        if self.action_loss_vlm_gradient_weight != 1.0:
            # Two-stage forward: prefix first, then suffix with gradient-weighted KV.
            prefix_att_2d_masks = full_att_2d[:, :prefix_len, :prefix_len]

            nvtx_push('paligemma_expert')
            (prefix_out, _), past_key_values = self.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
                fill_kv_cache=True,
                adarms_cond=[None, None],
            )

            _weight_kv_cache_gradients(
                past_key_values,
                self.action_loss_vlm_gradient_weight,
            )

            suffix_att_2d_masks = full_att_2d[:, prefix_len:, :]

            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=suffix_att_2d_masks,
                position_ids=suffix_position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=True,
                fill_kv_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            nvtx_pop()  # paligemma_expert
        else:
            full_position_ids = torch.cat([prefix_position_ids, suffix_position_ids], dim=1)

            nvtx_push('paligemma_expert')
            (prefix_out, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=full_att_2d,
                position_ids=full_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                fill_kv_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            nvtx_pop()  # paligemma_expert

        output = {}

        # Compute denoised output
        with nvtx_range('action_out_proj'):
            suffix_out = suffix_out[:, -self.n_action_steps :]
            suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
            v_t = self.action_out_proj(suffix_out, emb_ids=action_emb_ids)
            output['v_t'] = v_t

        # Compute language logits
        if self.enable_next_token_prediction:
            with nvtx_range('lang_head'):
                lang_out = prefix_out[:, -lang_length:, :]
                lang_out = lang_out.to(dtype=self.paligemma_with_expert.lm_head.weight.dtype)
                compact_masks = lang_loss_masks if self.compact_language_logits else None
                output.update(_compact_language_logits(self.paligemma_with_expert, lang_out, compact_masks))

        # Compute trajectory predictions
        if self.enable_learnable_traj_token:
            traj_out = prefix_out[:, traj_token_start_idx : traj_token_start_idx + self.num_traj_tokens, :]
            traj_out = traj_out.to(dtype=self.traj_out_proj.weight.dtype)
            traj_out, _ = self.traj_decoder(traj_out)
            traj_pred = self.traj_out_proj(traj_out)
            output['traj_pred'] = traj_pred

        return output

    def _forward_paligemma2_multi_flow_samples(
        self,
        prefix_embs: Tensor,
        prefix_pad_masks: Tensor,
        prefix_att_masks: Tensor,
        suffix_embs: Tensor,
        suffix_pad_masks: Tensor,
        suffix_att_masks: Tensor,
        adarms_cond: Tensor,
        action_emb_ids: Tensor,
        fast_action_indicator: Tensor | None,
        subtask_indicator: Tensor | None,
        lang_length: int,
        traj_token_start_idx: int | None,
        batch_mul: int,
    ) -> dict[str, Tensor]:
        """Run one PaliGemma2 prefix pass and K cached action suffixes."""
        prefix_batch_size, prefix_len = prefix_pad_masks.shape
        action_batch_size, suffix_len = suffix_pad_masks.shape
        expected_action_batch_size = prefix_batch_size * batch_mul
        if action_batch_size != expected_action_batch_size:
            raise ValueError(
                'The action batch must equal prefix batch * flow_matching_batch_mul, '
                f'but got {action_batch_size} != {prefix_batch_size} * {batch_mul}'
            )
        if action_emb_ids.shape[0] != action_batch_size:
            raise ValueError(f'action_emb_ids batch {action_emb_ids.shape[0]} does not match action batch {action_batch_size}')

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        if fast_action_indicator is not None and subtask_indicator is not None:
            prefix_att_2d_masks &= ~(fast_action_indicator[:, :, None] & subtask_indicator[:, None, :])
        prefix_position_ids = self._compute_1d_position_ids(prefix_pad_masks)

        (prefix_out, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            fill_kv_cache=True,
            adarms_cond=[None, None],
        )
        past_key_values = _repeat_kv_cache(
            past_key_values,
            repeats=batch_mul,
            gradient_weight=self.action_loss_vlm_gradient_weight,
        )

        action_prefix_pad_masks = prefix_pad_masks.repeat_interleave(batch_mul, dim=0)
        action_prefix_att_masks = prefix_att_masks.repeat_interleave(batch_mul, dim=0)
        action_fast_indicator = None if fast_action_indicator is None else fast_action_indicator.repeat_interleave(batch_mul, dim=0)
        action_subtask_indicator = None if subtask_indicator is None else subtask_indicator.repeat_interleave(batch_mul, dim=0)

        pad_masks_full = torch.cat([action_prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks_full = torch.cat([action_prefix_att_masks, suffix_att_masks], dim=1)
        full_att_2d = make_att_2d_masks(pad_masks_full, att_masks_full)
        full_att_2d = _mask_action_suffix_target_attention(
            full_att_2d,
            suffix_len,
            action_fast_indicator,
            action_subtask_indicator,
        )

        prefix_offsets = torch.sum(action_prefix_pad_masks, dim=-1, keepdim=True)
        suffix_position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        if action_fast_indicator is not None:
            suffix_position_ids = suffix_position_ids - action_fast_indicator.sum(dim=-1, keepdim=True)
        suffix_att_2d_masks = full_att_2d[:, prefix_len:, :]

        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=suffix_att_2d_masks,
            position_ids=suffix_position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=True,
            fill_kv_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        output = {}
        suffix_out = suffix_out[:, -self.n_action_steps :]
        suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        output['v_t'] = self.action_out_proj(suffix_out, emb_ids=action_emb_ids)

        if self.enable_next_token_prediction:
            lang_out = prefix_out[:, -lang_length:, :]
            lang_out = lang_out.to(dtype=self.paligemma_with_expert.lm_head.weight.dtype)
            output['lang_logits'] = self.paligemma_with_expert.language_out_proj(lang_out)

        if self.enable_learnable_traj_token:
            if traj_token_start_idx is None:
                raise ValueError('traj_token_start_idx is required when trajectory prediction is enabled')
            traj_out = prefix_out[:, traj_token_start_idx : traj_token_start_idx + self.num_traj_tokens, :]
            traj_out = traj_out.to(dtype=self.traj_out_proj.weight.dtype)
            traj_out, _ = self.traj_decoder(traj_out)
            output['traj_pred'] = self.traj_out_proj(traj_out)

        return output

    @staticmethod
    def _compute_1d_position_ids(pad_masks: Tensor) -> Tensor:
        """Build 1D RoPE position IDs from a padding mask (lingbot-vla / π0
        style).

        Equivalent to ``cumsum(pad_masks, dim=1) - 1``: each non-padding token
        gets a contiguous index, padding tokens reuse their predecessor's value
        (irrelevant since the attention mask masks them out anyway).

        Args:
            pad_masks: (B, L) bool/int padding mask, True/1 = real token.

        Returns:
            (B, L) long tensor of 1D positions.
        """
        return (torch.cumsum(pad_masks.long(), dim=1) - 1).clamp_min(0)

    def embed_prefix(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        lang_att_masks: Tensor | None = None,
        fast_action_indicator: Tensor | None = None,
        subtask_indicator: Tensor | None = None,
        proprioception: Tensor | None = None,
        agent_pos_mask: Tensor | None = None,
        proprioception_present: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor | None, int | None]:
        """Embeds prefix inputs (images, optional trajectory tokens, language)
        into a single sequence.

        This method uses a vision model (SigLIP) for images and an embedding layer
        for language tokens, combining them into a unified embedding sequence for
        the transformer.

        Args:
            images: A list of image tensors.
            img_masks: A list of boolean masks for the images.
            lang_tokens: Language input token IDs.
            lang_masks: Boolean mask for the language tokens.
            lang_att_masks: Optional attention masks for language tokens.
            fast_action_indicator: Optional tensor indicating fast action token positions.
            subtask_indicator: Optional tensor indicating subtask token positions.

        Returns:
            A tuple containing:
            - embs: The combined prefix embeddings.
            - pad_masks: The corresponding padding masks.
            - att_masks: The corresponding attention masks.
            - fast_action_indicator: Updated fast action indicator with image/traj prefix padding.
            - subtask_indicator: Updated subtask indicator with image/traj prefix padding.
            - traj_token_start_idx: The starting index of trajectory tokens, if they exist.
        """
        num_images = len(images)

        # Stack images and masks for batch processing
        images_stacked = torch.stack(images, dim=0)  # (num_images, bsize, ...)
        img_masks_stacked = torch.stack(img_masks, dim=0)  # (num_images, bsize)

        # Batch embed all images at once
        # Reshape to (num_images * bsize, ...)
        orig_shape = images_stacked.shape
        images_flat = images_stacked.reshape(-1, *orig_shape[2:])
        img_masks_flat = None
        if img_masks_stacked.ndim == 3:
            img_masks_flat = img_masks_stacked.reshape(-1, img_masks_stacked.shape[-1])
        with nvtx_range('vision_tower'):
            img_embs_flat = self.paligemma_with_expert.embed_image(images_flat, image_attention_mask=img_masks_flat)

        # Reshape back to (num_images, bsize, num_img_embs, emb_dim)
        bsize = orig_shape[1]
        img_embs = img_embs_flat.reshape(num_images, bsize, *img_embs_flat.shape[1:])

        # Normalize image embeddings
        img_emb_dim = img_embs.shape[-1]
        num_img_embs = img_embs.shape[2]

        if img_masks_stacked.ndim == 3:
            current_img_masks = img_masks_stacked[:, :, -1]
        else:
            current_img_masks = img_masks_stacked

        # Expand masks: (num_images, bsize) -> (num_images, bsize, num_img_embs)
        img_masks_expanded = current_img_masks[:, :, None].expand(num_images, bsize, num_img_embs)

        # Reshape to (bsize, num_images * num_img_embs, emb_dim)
        img_embs_concat = img_embs.transpose(0, 1).reshape(bsize, num_images * num_img_embs, img_emb_dim)
        img_masks_concat = img_masks_expanded.transpose(0, 1).reshape(bsize, num_images * num_img_embs)

        # Process language embeddings
        lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)
        lang_emb = lang_emb.to(dtype=img_embs_concat.dtype)
        lang_emb = self._scatter_proprioception_anchor(
            lang_tokens,
            lang_emb,
            proprioception,
            agent_pos_mask,
            proprioception_present,
        )

        num_lang_embs = lang_emb.shape[1]
        num_img_embs_total = num_images * num_img_embs
        num_traj_embs = self.num_traj_tokens if self.enable_learnable_traj_token else 0
        total_seq_len = num_img_embs_total + num_traj_embs + num_lang_embs

        # Pre-allocate final tensors
        embs = torch.empty(bsize, total_seq_len, img_emb_dim, dtype=img_embs_concat.dtype, device=img_embs_concat.device)
        pad_masks = torch.empty(bsize, total_seq_len, dtype=torch.bool, device=img_embs_concat.device)

        # Fill pre-allocated tensors
        embs[:, :num_img_embs_total] = img_embs_concat
        pad_masks[:, :num_img_embs_total] = img_masks_concat

        num_prefix_embs = num_img_embs_total
        traj_token_start_idx = None
        if self.enable_learnable_traj_token:
            traj_emb = self.traj_token[None].repeat(bsize, 1, 1)
            embs[:, num_img_embs_total : num_img_embs_total + num_traj_embs] = traj_emb
            pad_masks[:, num_img_embs_total : num_img_embs_total + num_traj_embs] = torch.ones(
                bsize, self.num_traj_tokens, dtype=torch.bool, device=traj_emb.device
            )
            traj_token_start_idx = num_prefix_embs
            num_prefix_embs += num_traj_embs

        embs[:, num_img_embs_total + num_traj_embs :] = lang_emb
        pad_masks[:, num_img_embs_total + num_traj_embs :] = lang_masks

        # Create attention masks (all zeros for full attention between image and language)
        att_masks = torch.zeros(total_seq_len, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(bsize, total_seq_len).clone()
        if lang_att_masks is not None:
            att_masks[:, -num_lang_embs:] = lang_att_masks
        if fast_action_indicator is not None:
            fast_action_indicator = F.pad(fast_action_indicator, (num_prefix_embs, 0))
        if subtask_indicator is not None:
            subtask_indicator = F.pad(subtask_indicator, (num_prefix_embs, 0))

        return embs, pad_masks, att_masks, fast_action_indicator, subtask_indicator, traj_token_start_idx

    def embed_suffix(self, noisy_actions: Tensor, timestep: Tensor, emb_ids: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Embeds suffix inputs (noisy actions and timestep) for the
        transformer.

        Args:
            noisy_actions: The noisy action tensor at the current timestep.
            timestep: The current timestep value.
            emb_ids: Embodiment IDs for selecting embodiment-specific layers.

        Returns:
            A tuple containing:
            - embs: The combined suffix embeddings.
            - pad_masks: The corresponding padding masks.
            - att_masks: The corresponding attention masks.
            - adarms_cond: The conditioning vector for AdaRMSNorm.
        """
        embs = []
        pad_masks = []
        att_masks = []

        if noisy_actions.ndim != 3:
            raise ValueError(f'noisy_actions must have shape [B, T, D], got {tuple(noisy_actions.shape)}')
        if noisy_actions.shape[1] != self.n_action_steps:
            raise ValueError(f'noisy_actions temporal length {noisy_actions.shape[1]} does not match ' f'policy.n_action_steps={self.n_action_steps}')

        action_emb = self.action_in_proj(noisy_actions, emb_ids=emb_ids)
        bsize, action_steps = action_emb.shape[:2]
        dtype = action_emb.dtype
        device = action_emb.device

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        with nvtx_range('time_mlp'):
            time_emb = create_sinusoidal_pos_embedding(timestep, self.proj_width, min_period=4e-3, max_period=4.0, device=device)
            time_emb = time_emb.type(dtype=dtype)

            # Time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = F.silu(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = F.silu(time_emb)
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_emb)

        action_time_mask = torch.ones(bsize, action_steps, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image and language do not attend to action tokens
        att_masks += [1] + ([0] * (action_steps - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def sample_noise(self, shape: tuple, device: torch.device | str) -> Tensor:
        """Samples noise from a standard normal distribution.

        Args:
            shape: The desired shape of the noise tensor.
            device: The device to create the tensor on.

        Returns:
            A tensor of the given shape filled with noise.
        """
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def set_action_denoise_mask(self, mask: Tensor | list[bool] | None) -> None:
        """Restrict flow denoising to configured action dimensions.

        The mask is applied to both the initial noise and every Euler update so padded action dimensions stay zero exactly as they did during masked-
        noise training. It is runtime-only and is not serialized into model config.
        """
        if mask is None:
            self._action_denoise_mask = None
            return

        resolved = torch.as_tensor(mask, dtype=torch.bool).reshape(-1)
        if resolved.numel() != self.max_action_dim:
            raise ValueError(f'action denoise mask must have {self.max_action_dim} entries, ' f'got {resolved.numel()}')
        self._action_denoise_mask = resolved

    def _mask_action_denoise_dims(self, value: Tensor) -> Tensor:
        mask = self._action_denoise_mask
        if mask is None:
            return value
        mask = mask.to(device=value.device).view(*([1] * (value.ndim - 1)), -1)
        return torch.where(mask, value, torch.zeros((), dtype=value.dtype, device=value.device))

    def _prepare_action_noise(
        self,
        noise: Tensor | None,
        batch_size: int,
        device: torch.device | str,
    ) -> Tensor:
        if noise is None:
            noise = self.sample_noise((batch_size, self.n_action_steps, self.max_action_dim), device)
        else:
            if noise.ndim != 3:
                raise ValueError(f'noise must have shape [B, T, D], got {tuple(noise.shape)}')
            if noise.shape[0] != batch_size:
                raise ValueError(f'noise batch size {noise.shape[0]} does not match input batch size {batch_size}')
            if noise.shape[1] != self.n_action_steps:
                raise ValueError(f'noise temporal length {noise.shape[1]} does not match policy.n_action_steps={self.n_action_steps}')
            if noise.shape[2] != self.max_action_dim:
                raise ValueError(f'noise action dimension {noise.shape[2]} does not match policy.max_action_dim={self.max_action_dim}')
        return self._mask_action_denoise_dims(noise)

    @torch.no_grad()
    def sample_actions(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        emb_ids: Tensor,
        enable_2d_traj_output: bool = False,
        noise: Tensor | None = None,
        lang_att_masks: Tensor | None = None,
        state_memory: Tensor | None = None,
        state_memory_masks: Tensor | None = None,
        proprioception: Tensor | None = None,
        agent_pos_mask: Tensor | None = None,
        proprioception_present: Tensor | None = None,
        state: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Performs a full inference pass to sample actions.

        This involves creating a key-value cache from image and language inputs,
        then iteratively denoising an initial noise tensor to produce the final actions.

        Args:
            images: A list of image tensors.
            img_masks: A list of boolean masks for the images.
            lang_tokens: Language input token IDs.
            lang_masks: Boolean mask for the language tokens.
            emb_ids: Embodiment IDs for selecting embodiment-specific layers.
            enable_2d_traj_output: If True, also returns trajectory predictions.
            noise: Optional initial noise tensor. If None, it will be sampled.

        Returns:
            - The sampled action tensor.
            - If `enable_2d_traj_output` is True, a tuple of the action tensor
              and the trajectory prediction tensor.
        """
        bsize = images[0].shape[0]
        device = images[0].device

        noise = self._prepare_action_noise(noise, bsize, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks, _, _, traj_token_start_idx = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            lang_att_masks=lang_att_masks,
            proprioception=proprioception,
            agent_pos_mask=agent_pos_mask,
            proprioception_present=proprioception_present,
        )
        prefix_embs, prefix_pad_masks, prefix_att_masks, _, _, state_pad_masks = self._prepend_proprio_prefix_tokens(
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            state_memory,
            state_memory_masks,
            emb_ids,
        )
        if state_pad_masks is not None and traj_token_start_idx is not None:
            traj_token_start_idx += state_pad_masks.shape[1]
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = self._compute_1d_position_ids(prefix_pad_masks)

        # Compute image and language key value cache
        (prefix_out, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.use_cache,
            fill_kv_cache=True,
            adarms_cond=[None, None],
        )

        if enable_2d_traj_output:
            traj_out = prefix_out[:, traj_token_start_idx : traj_token_start_idx + self.num_traj_tokens, :]
            traj_out = traj_out.to(dtype=self.traj_out_proj.weight.dtype)
            traj_out, _ = self.traj_decoder(traj_out)
            traj_pred = self.traj_out_proj(traj_out)

        x_t = noise.to(dtype=self.action_in_proj.weight.dtype)
        dt = -1.0 / self.num_steps
        timesteps = torch.arange(1.0, -dt / 2, dt, dtype=torch.float32, device=device)
        for timestep in timesteps:
            v_t = self.denoise_step(
                prefix_pad_masks,
                past_key_values,
                x_t,
                timestep.expand(bsize),
                emb_ids,
            )
            x_t = self._mask_action_denoise_dims(x_t + dt * v_t)

        if enable_2d_traj_output:
            return x_t, traj_pred

        return x_t

    # ======================================================================
    #                        Eval / inference entry
    # ======================================================================
    def configure_eval(
        self,
        *,
        processor_path: str,
        resize_imgs_with_padding: tuple[int, int] | list[int] | None = None,
        present_img_keys: list[str] | None = None,
        max_length: int | None = None,
        discrete_state_input: bool | None = None,
        prompt_cfg: dict | None = None,
    ) -> 'GigaBrain07Policy':
        """Configure the preprocessing used by :meth:`predict_action`.

        An eval factory calls this once after building / loading the model so
        ``predict_action`` can lazily build the image + prompt transforms that
        mirror training preprocessing. Resetting the cached transforms lets a
        caller reconfigure (e.g. a different camera layout) between runs.

        Args:
            processor_path: Directory of the PaliGemma2 tokenizer / processor
                (same checkpoint the model was trained against).
            resize_imgs_with_padding: ``(width, height)`` resize-with-pad target
                for every camera view. Defaults to ``(224, 224)``.
            present_img_keys: Ordered camera keys. Eval views are mapped onto
                these in order; missing views become masked zero images, exactly
                as in training. Defaults to the GigaBrain 3-camera layout.
            max_length: Prompt token budget (padding length). Defaults to 256.
            discrete_state_input: Whether the ``prompt`` state mode folds the
                discretized state into the text prompt. Defaults to True.
            prompt_cfg: Optional explicit override dict forwarded verbatim to
                ``PromptTokenizerTransform`` (wins over the derived defaults).
        """
        self._processor_path = str(processor_path)
        if resize_imgs_with_padding is not None:
            self._eval_resize_imgs_with_padding = tuple(int(v) for v in resize_imgs_with_padding)
        if present_img_keys is not None:
            self._eval_present_img_keys = list(present_img_keys)
        if max_length is not None:
            self._eval_max_length = int(max_length)
        if discrete_state_input is not None:
            self._eval_discrete_state_input = bool(discrete_state_input)
        if prompt_cfg is not None:
            self._eval_prompt_cfg = dict(prompt_cfg)
        # Invalidate caches so the next predict_action rebuilds with new config.
        self._image_transform = None
        self._prompt_tokenizer_transform = None
        return self

    def _get_eval_image_transform(self):
        """Lazily build (and cache) the GigaBrain image preprocessing core.

        Uses ``ImageTransform`` from the vendored ``giga_models`` package (the
        same class the training transform wraps), so ``predict_action`` reuses
        the exact resize-with-pad + ``[-1, 1]`` SigLIP normalization without
        pulling in the LoongForge training-side data stack.
        """
        cached = getattr(self, '_image_transform', None)
        if cached is None:
            from giga_models.pipelines.vla.giga_brain_0.giga_brain_0_utils import ImageTransform

            size = self._eval_resize_imgs_with_padding or (224, 224)
            cached = ImageTransform(
                is_train=False,
                resize_imgs_with_padding=tuple(size),
                present_img_keys=self._eval_present_img_keys,
                enable_image_aug=False,
                vlm_type=self.vlm_type,
            )
            self._image_transform = cached
        return cached

    def _get_eval_prompt_tokenizer(self):
        """Lazily build (and cache) the GigaBrain prompt tokenizer core.

        Mirrors the training ``PromptTokenizerTransform`` but fixed to the
        inference regime (``is_train=False``, no FAST-action / subtask targets),
        so the emitted ``lang_tokens`` / ``lang_masks`` / ``lang_att_masks``
        match the training prefix layout the flow-matching backbone expects.
        """
        cached = getattr(self, '_prompt_tokenizer_transform', None)
        if cached is None:
            if not self._processor_path:
                raise RuntimeError(
                    'predict_action requires a tokenizer/processor path; set it via '
                    'configure_eval(processor_path=...) or assign model._processor_path.'
                )
            from giga_models.pipelines.vla.giga_brain_0.giga_brain_0_utils import PromptTokenizerTransform

            cfg = dict(self._eval_prompt_cfg or {})
            cfg.setdefault('is_train', False)
            cfg.setdefault('tokenizer_model_path', self._processor_path)
            # fast_tokenizer_path is only touched when FAST actions are encoded;
            # flow-matching inference never triggers that lazy load.
            cfg.setdefault('fast_tokenizer_path', '')
            cfg.setdefault('max_length', self._eval_max_length)
            cfg.setdefault('discrete_state_input', self._eval_discrete_state_input)
            cfg.setdefault('encode_action_input', False)
            cfg.setdefault('encode_sub_task_input', False)
            cfg.setdefault('autoregressive_inference_mode', False)
            cfg.setdefault('vlm_type', self.vlm_type)
            cfg.setdefault('state_input_mode', self.state_input_mode)
            cached = PromptTokenizerTransform(**cfg)
            self._prompt_tokenizer_transform = cached
        return cached

    @staticmethod
    def _coerce_eval_view_to_chw(view) -> Tensor:
        """Coerce one eval image view to a ``[C, H, W]`` tensor for ImageTransform.

        Accepts a CHW/HWC torch tensor, a HWC ndarray, or a PIL image. ``uint8``
        / ``[0, 255]`` pixels are kept as ``uint8`` (ImageTransform divides by
        255 then rescales to ``[-1, 1]``); already-float ``[0, 1]`` pixels are
        passed through unchanged.
        """
        import numpy as np

        if 'PIL' in str(type(view)):
            view = np.asarray(view)
        if isinstance(view, np.ndarray):
            view = torch.as_tensor(view)
        elif not isinstance(view, torch.Tensor):
            view = torch.as_tensor(np.asarray(view))

        if view.ndim == 2:  # grayscale H,W -> H,W,3
            view = view.unsqueeze(-1).expand(-1, -1, 3)
        if view.ndim != 3:
            raise ValueError(f'predict_action: each image view must be 2D/3D, got shape {tuple(view.shape)}')

        # HWC -> CHW when the trailing axis is the channel axis.
        if view.shape[0] not in (1, 3, 4) and view.shape[-1] in (1, 3, 4):
            view = view.permute(2, 0, 1)
        view = view.contiguous()

        if view.is_floating_point():
            # Values clearly outside [0, 1] are byte-range floats: hand them to
            # ImageTransform as uint8 so its /255 branch does the right thing.
            if torch.isfinite(view).any() and float(view.max()) > 1.5:
                view = view.round().clamp_(0, 255).to(torch.uint8)
        elif view.dtype != torch.uint8:
            view = view.clamp_(0, 255).to(torch.uint8)
        return view

    def _resolve_eval_stats(self, dataset_stats, field: str, emb_key: int):
        """Resolve normalization stats for ``field`` into a Normalize-ready dict.

        ``dataset_stats[field]`` may be either a flat ``{'q01':.., 'q99':..}`` /
        ``{'mean':.., 'std':..}`` block (single dataset) or an already
        embodiment-keyed ``{emb_id: {...}}`` mapping. Returns
        ``(stats_by_key, use_quantiles)`` or ``(None, False)`` when unavailable.
        """
        if not dataset_stats:
            return None, False
        # LoongForge norm-stats files wrap the per-field blocks under a top-level
        # ``norm_stats`` key (e.g. agibot_a2d_norm_stats.json); unwrap it so the
        # dataset_statistics_path file feeds normalization directly.
        if 'norm_stats' in dataset_stats and isinstance(dataset_stats['norm_stats'], dict):
            dataset_stats = dataset_stats['norm_stats']
        raw = dataset_stats.get(field)
        if not isinstance(raw, dict) or not raw:
            return None, False
        if 'q01' in raw or 'mean' in raw:
            stats = {emb_key: raw}
        else:
            stats = raw
        sample = next(iter(stats.values()))
        if not isinstance(sample, dict):
            return None, False
        use_quantiles = 'q01' in sample and 'q99' in sample
        if not use_quantiles and not ('mean' in sample and 'std' in sample):
            return None, False
        return stats, use_quantiles

    @torch.no_grad()
    def predict_action(
        self,
        images,
        instructions,
        state=None,
        dataset_stats=None,
        emb_id=None,
        **kwargs,
    ):
        """Eval-facing entry point matching the shared ``predict_action`` interface.

        Mirrors ``XVLAPolicy`` / ``PI05Policy``: it owns model-specific
        preprocessing (image resize+normalize, prompt tokenization), state
        normalization, flow-matching action sampling, and action
        unnormalization. Dataset-specific action repacking (delta / quaternion→
        6D) stays in the eval-side ActionDecoder, not here.

        Args:
            images: Batched images. Either a length-``B`` list of per-sample view
                lists (each view a CHW/HWC tensor, HWC ndarray, or PIL image), or
                a flat length-``B`` list of single-view images (auto-wrapped).
            instructions: ``List[str]`` of language instructions (length ``B``).
            state: Optional proprio state: ``None``, a 1-D vector, or a
                ``[B, D_state]`` tensor / ndarray / list.
            dataset_stats: Optional normalization stats. ``dataset_stats['action']``
                (and ``['observation.state']``) may be a flat q01/q99 or mean/std
                block, or an embodiment-keyed mapping. When action stats are
                present the sampled actions are unnormalized; otherwise the raw
                model-space actions are returned.
            emb_id: Embodiment id (int, or ``[B]`` ids). Defaults to 0.

        Returns:
            ``np.ndarray`` of shape ``[B, n_action_steps, max_action_dim]``.
        """
        import numpy as np

        del kwargs  # extra eval payload keys (unnorm_key, cfg_scale, ...) are ignored
        self.eval()
        device = next(self.parameters()).device

        if not isinstance(images, (list, tuple)) or len(images) == 0:
            raise ValueError('predict_action: images must be a non-empty list.')
        if not isinstance(images[0], (list, tuple)):
            images = [[img] for img in images]
        batch_size = len(images)

        if isinstance(instructions, str):
            instructions = [instructions]
        if not isinstance(instructions, (list, tuple)) or len(instructions) != batch_size:
            raise ValueError(
                f'predict_action: got {batch_size} image samples but '
                f'{len(instructions) if hasattr(instructions, "__len__") else "?"} instructions.'
            )

        # Embodiment ids → [B] long tensor.
        if emb_id is None:
            emb_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
        elif isinstance(emb_id, torch.Tensor):
            emb_ids = emb_id.reshape(-1).to(device=device, dtype=torch.long)
            if emb_ids.numel() == 1:
                emb_ids = emb_ids.expand(batch_size).contiguous()
        else:
            emb_ids = torch.full((batch_size,), int(emb_id), dtype=torch.long, device=device)
        emb_key = int(emb_ids[0].item())

        image_tf = self._get_eval_image_transform()
        prompt_tf = self._get_eval_prompt_tokenizer()
        present_keys = list(image_tf.present_img_keys)

        # Optional state normalization (shared across the prompt + proprio paths).
        state_stats, state_use_q = self._resolve_eval_stats(dataset_stats, 'observation.state', emb_key)
        state_normalize = None
        if state_stats is not None:
            from giga_models.pipelines.vla.giga_brain_0.giga_brain_0_utils import Normalize

            state_normalize = Normalize(state_stats, use_quantiles=state_use_q)

        def _prep_state_vec(idx: int) -> Tensor | None:
            if state is None:
                return None
            s = state
            if not isinstance(s, torch.Tensor):
                s = torch.as_tensor(np.asarray(s), dtype=torch.float32)
            s = s.to(dtype=torch.float32)
            if s.dim() == 1:
                s = s.unsqueeze(0).expand(batch_size, -1)
            if s.shape[0] != batch_size:
                raise ValueError(f'predict_action: state batch {s.shape[0]} != image batch {batch_size}.')
            vec = s[idx]
            if state_normalize is not None:
                vec = state_normalize(vec, embodiment_id=emb_key)
            if vec.shape[-1] < self.max_state_dim:
                vec = F.pad(vec, (0, self.max_state_dim - vec.shape[-1]))
            elif vec.shape[-1] > self.max_state_dim:
                vec = vec[..., : self.max_state_dim]
            return vec

        # Per-sample preprocessing, then stack to batched model inputs.
        per_sample_imgs: list[list[Tensor]] = []
        per_sample_masks: list[list[Tensor]] = []
        lang_tokens_rows: list[Tensor] = []
        lang_masks_rows: list[Tensor] = []
        lang_att_rows: list[Tensor] = []
        proprio_rows: list[Tensor] = []

        for i in range(batch_size):
            views = images[i]
            if len(views) > len(present_keys):
                raise ValueError(
                    f'predict_action: sample {i} has {len(views)} views but the configured '
                    f'camera layout only defines {len(present_keys)} keys ({present_keys}).'
                )
            data: dict = {}
            for v, view in enumerate(views):
                data[present_keys[v]] = self._coerce_eval_view_to_chw(view)

            state_vec = _prep_state_vec(i)
            if state_vec is not None:
                data['observation.state'] = state_vec
            data['task'] = str(instructions[i])
            data['embodiment_id'] = emb_key

            imgs_i, masks_i, _ = image_tf(data)
            per_sample_imgs.append([t.to(device) for t in imgs_i])
            per_sample_masks.append([m.to(device) for m in masks_i])

            lang_ids, lang_mask, lang_att, _loss_mask, _fast, _subtask, _subtask_only = prompt_tf(data)
            lang_tokens_rows.append(lang_ids)
            lang_masks_rows.append(lang_mask)
            lang_att_rows.append(lang_att)
            if self.state_input_mode == 'proprio_anchor':
                proprio_rows.append(state_vec if state_vec is not None else torch.zeros(self.max_state_dim))

        num_cameras = len(per_sample_imgs[0])
        images_list = [
            torch.stack([per_sample_imgs[b][c] for b in range(batch_size)], dim=0)
            for c in range(num_cameras)
        ]
        img_masks = [
            torch.stack([per_sample_masks[b][c].reshape(()) for b in range(batch_size)], dim=0).to(torch.bool)
            for c in range(num_cameras)
        ]

        lang_tokens = torch.stack(lang_tokens_rows, dim=0).to(device=device, dtype=torch.long)
        lang_masks = torch.stack(lang_masks_rows, dim=0).to(device=device, dtype=torch.bool)
        lang_att_masks = torch.stack(lang_att_rows, dim=0).to(device=device, dtype=torch.bool)

        sample_kwargs: dict = {
            'images': images_list,
            'img_masks': img_masks,
            'lang_tokens': lang_tokens,
            'lang_masks': lang_masks,
            'emb_ids': emb_ids,
            'lang_att_masks': lang_att_masks,
        }

        # Wire the continuous-state paths when the model expects them. The
        # 'prompt' mode already folded the discretized state into lang_tokens.
        if self.state_input_mode == 'proprio_memory' and state is not None:
            state_mem = torch.stack(
                [(_prep_state_vec(i)) for i in range(batch_size)], dim=0
            ).to(device=device, dtype=torch.float32)
            # [B, D] -> [B, K, D] with K = observation_memory_size (repeat current frame).
            state_mem = state_mem[:, None, :].expand(batch_size, self.observation_memory_size, self.max_state_dim).contiguous()
            sample_kwargs['state_memory'] = state_mem
            sample_kwargs['state_memory_masks'] = torch.ones(
                batch_size, self.observation_memory_size, dtype=torch.bool, device=device
            )
        elif self.state_input_mode == 'proprio_anchor':
            propri = torch.stack(proprio_rows, dim=0).to(device=device, dtype=torch.float32)
            propri = propri[..., : self.propri_dim]
            sample_kwargs['proprioception'] = propri[:, None, :]
            sample_kwargs['agent_pos_mask'] = torch.ones(
                batch_size, 1, self.propri_dim, dtype=torch.float32, device=device
            )
            sample_kwargs['proprioception_present'] = torch.ones(batch_size, dtype=torch.bool, device=device)

        actions = self.sample_actions(**sample_kwargs)
        if isinstance(actions, tuple):
            actions = actions[0]
        actions = actions.to(dtype=torch.float32)

        action_stats, action_use_q = self._resolve_eval_stats(dataset_stats, 'action', emb_key)
        if action_stats is not None:
            from giga_models.pipelines.vla.giga_brain_0.giga_brain_0_utils import Unnormalize

            unnormalize = Unnormalize(action_stats, use_quantiles=action_use_q).to(actions.device)
            actions = unnormalize(actions, embodiment_id=emb_key)

        return actions.cpu().numpy()

    def denoise_step(
        self,
        prefix_pad_masks: Tensor,
        past_key_values: dict,
        x_t: Tensor,
        timestep: Tensor,
        emb_ids: Tensor,
    ) -> Tensor:
        """Applies one denoising step to `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep, emb_ids=emb_ids)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.use_cache,
            fill_kv_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.n_action_steps :]
        suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        v_t = self.action_out_proj(suffix_out, emb_ids=emb_ids)
        return v_t

    @torch.no_grad()
    def init_lang_generation(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        image_grid_thw: Tensor | None = None,
        lang_att_masks: Tensor | None = None,
        emb_ids: Tensor | None = None,
        state_memory: Tensor | None = None,
        state_memory_masks: Tensor | None = None,
        proprioception: Tensor | None = None,
        agent_pos_mask: Tensor | None = None,
        proprioception_present: Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Prefill stage: build KV cache from images + existing language prefix and
        return logits for the next token.

        ``lang_att_masks`` uses make_att_2d_masks semantics: 1 starts a new
        causal block and 0 continues the bidirectional block.

        Returns:
            A tuple containing:
            - next_logits: Logits for the next token prediction, shape (batch_size, vocab_size).
            - state: A dictionary containing the state for autoregressive generation:
                - 'past_key_values': The key-value cache from the transformer.
                - 'position_cursor': The current position in the sequence for position embeddings.
                - 'batch_size': The batch size.
                - 'device': The device of the tensors.
                - 'prefix_pad_masks': The padding mask for the prefix.
        """
        bsize = lang_tokens.shape[0]
        device = lang_tokens.device

        # embed prefix (images + lang)
        prefix_embs, prefix_pad_masks, prefix_att_masks, _, _, _ = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            lang_att_masks=lang_att_masks,
            proprioception=proprioception,
            agent_pos_mask=agent_pos_mask,
            proprioception_present=proprioception_present,
        )
        prefix_embs, prefix_pad_masks, prefix_att_masks, _, _, _ = self._prepend_proprio_prefix_tokens(
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            state_memory,
            state_memory_masks,
            emb_ids,
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = self._compute_1d_position_ids(prefix_pad_masks)

        # forward to build cache and get language hidden states
        (prefix_out, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            fill_kv_cache=True,
            adarms_cond=[None, None],
        )

        # Take the hidden state of the last valid language token per sample as
        # the condition for the next-token distribution
        lang_length = lang_masks.shape[-1]
        lang_region = prefix_out[:, -lang_length:, :]
        batch_indices = torch.arange(bsize, device=device)
        last_hidden = lang_region[batch_indices, -1, :].unsqueeze(1)
        last_hidden = last_hidden.to(dtype=self.paligemma_with_expert.lm_head.weight.dtype)
        next_logits = self.paligemma_with_expert.language_out_proj(last_hidden)[:, 0, :]

        # Next-token position (0-indexed) equals the number of consumed tokens so far
        position_cursor = torch.sum(prefix_pad_masks, dim=-1).to(dtype=torch.long)

        state = {
            'past_key_values': past_key_values,
            'position_cursor': position_cursor,
            'batch_size': bsize,
            'device': device,
            'prefix_pad_masks': prefix_pad_masks[:, None, :],
        }
        if self.state_input_mode == 'proprio_memory':
            state = self._attach_full_prefill_generation_state(
                state,
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                lang_att_masks,
                image_grid_thw,
                emb_ids,
                state_memory,
                state_memory_masks,
            )
        return next_logits, state

    @torch.no_grad()
    def next_lang_logits(
        self,
        state: dict,
        input_token: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Decode step: given the previously generated token, return logits for
        the next token and update the KV cache/state.

        Args:
            state: A dictionary containing the state from a previous generation step.
                   See `init_lang_generation` for details on its contents.
            input_token: The token generated in the previous step, shape (batch_size, 1).

        Returns:
            A tuple containing:
            - next_logits: Logits for the next token prediction, shape (batch_size, vocab_size).
            - state: The updated state dictionary for the next generation step.
        """
        if state.get('full_prefill_decode'):
            return self._next_lang_logits_full_prefill(state, input_token)
        past_key_values = state['past_key_values']
        position_cursor = state['position_cursor']
        bsize = state['batch_size']
        device = state['device']
        prefix_pad_masks = state['prefix_pad_masks']

        # Prepare single-step language token embedding
        if input_token.dtype != torch.long:
            input_token = input_token.to(torch.long)
        token_emb = self.paligemma_with_expert.embed_language_tokens(input_token)
        # Match scaling used in embed_prefix to keep prefill/decoding consistent
        lang_emb_dim = token_emb.shape[-1]
        token_emb = token_emb * math.sqrt(lang_emb_dim)

        # Build attention mask: one-step query attends to all history and itself
        att_2d_masks = torch.cat([prefix_pad_masks, torch.ones((bsize, 1, 1), dtype=torch.bool, device=device)], dim=2)

        # Position id is the current cursor (0-indexed)
        position_ids = position_cursor.view(bsize, 1).to(device)

        (lang_out, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[token_emb, None],
            use_cache=True,
            fill_kv_cache=False,
            adarms_cond=[None, None],
            auto_regression_inference_mode=True,
        )

        # Single step only, shape: (b, 1, d)
        lang_out = lang_out[:, -1:, :]
        lang_out = lang_out.to(dtype=self.paligemma_with_expert.lm_head.weight.dtype)
        next_logits = self.paligemma_with_expert.language_out_proj(lang_out)[:, 0, :]

        # Advance the cursor; KV cache is persistently appended inside the model
        state['past_key_values'] = past_key_values
        state['position_cursor'] = position_cursor + 1
        state['prefix_pad_masks'] = att_2d_masks
        return next_logits, state


class EmbodimentSpecificLinear(nn.Module):
    """A linear layer with weights and biases specific to an embodiment
    category."""

    def __init__(self, input_dim: int, output_dim: int, num_categories: int = 1, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.num_categories = num_categories
        self.weight = nn.Parameter(torch.empty(num_categories, input_dim, output_dim, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(num_categories, output_dim, dtype=dtype))

        # Initialize using Linear's default initialization: U(-k, k) where k = 1/sqrt(in_features)
        k = 1.0 / math.sqrt(input_dim)
        nn.init.uniform_(self.weight, -k, k)
        nn.init.uniform_(self.bias, -k, k)

    def forward(self, x: Tensor, emb_ids: Tensor) -> Tensor:
        """Applies the embodiment-specific linear transformation.

        Args:
            x: The input tensor.
            emb_ids: A tensor of embodiment IDs, used to select the appropriate
                     weights and biases.

        Returns:
            The transformed tensor.
        """
        # Use one-hot to aggregate per-category weights/biases to avoid dynamic indexing (compile-friendly)
        if emb_ids.dtype != torch.long:
            emb_ids = emb_ids.to(torch.long)
        emb_ids = emb_ids.reshape(-1).to(device=self.weight.device)

        one_hot = F.one_hot(emb_ids, num_classes=self.num_categories).to(dtype=self.weight.dtype)

        selected_weight = torch.einsum('bc,cij->bij', one_hot, self.weight)
        selected_bias = torch.einsum('bc,cj->bj', one_hot, self.bias)

        return torch.bmm(x, selected_weight) + selected_bias.unsqueeze(1)

    def extra_repr(self):
        return f'EmbodimentSpecificLinear(num_categories={self.num_categories}, input_dim={self.weight.shape[1]}, output_dim={self.weight.shape[2]})'


def get_safe_dtype(dtype: torch.dtype, device: str | torch.device):
    """MPS is currently not compatible with float64."""
    if isinstance(device, torch.device):
        device = device.type
    if device == 'mps' and dtype == torch.float64:
        return torch.float32
    else:
        return dtype


def create_sinusoidal_pos_embedding(time: torch.Tensor, dimension: int, min_period: float, max_period: float, device='cpu') -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar
    positions."""
    if dimension % 2 != 0:
        raise ValueError(f'dimension ({dimension}) must be divisible by 2')

    if time.ndim != 1:
        raise ValueError('The time tensor is expected to be of shape `(batch_size, )`.')

    dtype = get_safe_dtype(torch.float64, device)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def make_att_2d_masks(pad_masks: Tensor, att_masks: Tensor) -> Tensor:
    """Creates a 2D attention mask from 1D padding and attention-type masks.

    This function is copied from big_vision.

    The `att_masks` allow for different attention patterns:
    - `[[1, 1, 1, 1, 1, 1]]`: Purely causal attention.
    - `[[0, 0, 0, 1, 1, 1]]`: Prefix-LM attention. The first 3 tokens attend
      to each other, and the last 3 tokens are causal.
    - `[[1, 0, 1, 0, 1, 0]]`: Causal attention between blocks. Tokens can
      attend to previous blocks and tokens within the same block.

    Args:
        pad_masks: bool[B, N] indicating valid (true) vs. padding (false) tokens.
        att_masks: int[B, N] defining attention type. A `1` at a position
                   indicates the start of a new causal block.

    Returns:
        A 2D boolean attention mask of shape (B, N, N).
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    # att_masks arrive as bf16 under mixed-precision training (embed_suffix builds them
    # with embs.dtype); bf16 has a 7-bit mantissa, so cumsum values above 256 round to
    # even integers and the block comparison below corrupts the mask for fully-causal
    # prefixes longer than 256 real tokens (train-only; inference builds masks in long).
    cumsum = torch.cumsum(att_masks.long(), dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks
