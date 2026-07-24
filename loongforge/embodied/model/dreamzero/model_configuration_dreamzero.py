# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the DreamZero Wan-DiT VLA policy.

DreamZero supports two backbone variants — both are CausalWanModel (chunked
causal attention) — and multiple embodiments. Backbone hyperparameters match
the DreamZero Wan2.1-I2V-14B and Wan2.2-TI2V-5B action-head recipes.

Runtime notes:
- VAE/T5/CLIP encoders run inside ``ActionHead.forward()`` unless the structured
  ``precomputed_cache`` model section supplies cached features.
- Flow matching uses DreamZero's scheduler implementation, not
  ``loongforge/models/diffusion/wan/wan_flow_match.py``.
- Megatron-style fields (``num_layers`` / ``hidden_size`` / etc.) are populated
  in ``__post_init__`` from ``backbone_variant`` for shared trainer utilities.
"""

from dataclasses import dataclass, field
from typing import Any

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature


# Backbone hyperparameters for the supported DreamZero variants.
_BACKBONE_PRESETS: dict[str, dict] = {
    "wan21_14b": {
        # Wan2.1-I2V-14B-480P action-head recipe.
        "model_type": "i2v",
        "dim": 5120,
        "in_dim": 36,
        "ffn_dim": 13824,
        "out_dim": 16,
        "freq_dim": 256,
        "num_heads": 40,
        "num_layers": 40,
        "frame_seqlen": 880,
        "concat_first_frame_latent": True,
        "vae_class": "WanVideoVAE",
        "vae_z_dim": 16,
        "vae_dim": 96,
        "target_video_height": None,  # Wan2.1 keeps native resolution
        "target_video_width": None,
    },
    "wan22_5b": {
        # Wan2.2-TI2V-5B action-head recipe.
        "model_type": "ti2v",
        "dim": 3072,
        "in_dim": 48,
        "ffn_dim": 14336,
        "out_dim": 48,
        "freq_dim": 256,
        "num_heads": 24,
        "num_layers": 30,
        "frame_seqlen": 50,
        "concat_first_frame_latent": False,
        "vae_class": "WanVideoVAE38",
        "vae_z_dim": 48,
        "vae_dim": 160,
        "target_video_height": 160,
        "target_video_width": 320,
    },
}


@dataclass
class DreamZeroConfig(PreTrainedConfig):
    """DreamZero policy hyperparameters and Megatron-compatible training knobs."""

    model_type: str = "dreamzero"

    # ------------------------------------------------------------------
    # Backbone selection
    # ------------------------------------------------------------------
    backbone_variant: str = "wan22_5b"  # one of {"wan21_14b", "wan22_5b"}
    model_dtype: str = "bfloat16"

    # ------------------------------------------------------------------
    # Action head (from wan_flow_matching_action_tf.yaml)
    # ------------------------------------------------------------------
    num_frames: int = 49
    action_horizon: int = 48
    n_action_steps: int = 48
    state_horizon: int = 1
    text_len: int = 512

    # state/action padding (multi-embodiment; per-embodiment shape filled by
    # modality_config; here we cap to the largest dimension in {droid:8, agibot:32, yam:14}.)
    max_state_dim: int = 32
    max_action_dim: int = 32

    # WANPolicyHead hidden dim, separate from both the DiT backbone hidden size
    # and CausalWanModel's internal state/action MLP hidden dim.
    add_pos_embed: bool = True
    action_head_hidden_size: int = 64
    # CausalWanModel keeps its own state/action hidden size; do not derive it
    # from the backbone hidden size.
    dit_action_state_hidden_size: int = 1024
    attn_dropout: float = 0.2
    input_embedding_dim: int = 1536

    # Chunked causal mask shape (CausalWanModel default)
    num_frame_per_block: int = 1
    num_action_per_block: int = 32
    num_state_per_block: int = 1
    max_chunk_size: int = 49
    # Attention backend selector for DreamZero causal self-attention.
    attention_backend: str = "FA2"  # {"FA2", "FA3", "torch", "TE", "torch_onnx"}

    # ------------------------------------------------------------------
    # Flow-matching scheduler
    # ------------------------------------------------------------------
    num_inference_timesteps: int = 4  # not used during training
    # ``1`` selects the fallback all-True inference skip mask, so all diffusion
    # steps run the DiT.
    num_dit_steps: int = 1
    # CUDA backend flags are process-global; expose this only through the
    # DreamZero config path that needs it.
    disable_reduced_precision_reduction: bool = True
    # DreamZero uses explicit inner autocast blocks in the action head; avoid an
    # additional trainer-level autocast around the whole model call.
    disable_train_autocast: bool = True
    skip_init_weights: bool = True
    # Round optimizer-visible trainable parameters once before the first train
    # step when a recipe needs exact dtype placement.
    fsdp_initial_param_rounding_dtype: str | None = "bf16"

    # ------------------------------------------------------------------
    # Performance switches
    # ------------------------------------------------------------------
    # Self-attention compile is part of the DreamZero default path. Other
    # switches stay disabled unless a recipe enables them explicitly.
    compile_causal_attention: bool = True
    compile_causal_attention_parts: str = "clean,state,noisy_image,noisy_action"
    compile_causal_attention_warmup_frames: str = ""
    compile_causal_attention_warmup_blocks: str = "all"
    compile_causal_attention_warmup_batch: int = 1
    compile_causal_attention_warmup_backward: bool = False
    compile_causal_cross_attention: bool = False
    compile_cross_attention_emulate_precision_casts: bool = False
    compile_block_norm_modulate: bool = False
    block_norm_modulate_impl: str = "compile"
    block_norm_modulate_triton_warps: int = 8
    qk_rmsnorm_impl: str = "wan"  # {"wan", "te"}
    skip_single_state_attention: bool = False
    manual_self_attn_linear_backward: bool = False
    flash_attention_dense: bool = False
    flash_attention_dense_min_q: int = 0
    flash_attention_dense_policy: str = "legacy_min_q"
    cache_fa_lens: bool = False
    cache_fa_lens_clone: bool = False
    avoid_rope_reconcat: bool = False
    fused_rope: bool = False
    fused_rope_fp64: bool = False
    batch_vae_encode: bool = False
    prompt_emb_cache: str = ""  # {"", "gpu", "cpu"}
    prompt_emb_cache_max_entries: int = 128
    # Structured offline sample-level artifact cache. This is the only public
    # config surface for precomputed video, first-frame, and prompt features.
    precomputed_cache: Any = None

    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000
    decouple_video_action_noise: bool = False
    video_noise_beta_alpha: float = 3.0
    video_noise_beta_beta: float = 1.0

    # ------------------------------------------------------------------
    # External encoders (frozen; default loaded from HF)
    # ------------------------------------------------------------------
    text_encoder_pretrained_path: str = ""   # google/umt5-xxl
    image_encoder_pretrained_path: str = ""  # CLIP (Wan2.1 only)
    vae_pretrained_path: str = ""
    # Directory containing diffusion_pytorch_model*.safetensors[.index.json].
    # When non-empty, dreamzero_provider loads DiT weights into CausalWanModel
    # via the diffusers-to-Civitai rename table.
    dit_pretrained_path: str = ""
    # Optional full DreamZero checkpoint containing the complete
    # CausalWanModel under action_head.model.*. This fills the Wan2.2 release
    # gaps such as image-conditioning cross-attn/img_emb tensors before
    # action/state compatibility loading.
    dit_init_checkpoint_path: str = ""
    # Optional full DreamZero checkpoint containing action/state/action_decoder
    # tensors absent from the base Wan DiT checkpoint. When omitted, these
    # tensors keep their normal random initialization.
    action_state_init_checkpoint_path: str = ""

    # VAE encode parameters.
    vae_tiled: bool = False
    vae_tile_size_height: int = 34
    vae_tile_size_width: int = 34
    vae_tile_stride_height: int = 18
    vae_tile_stride_width: int = 16

    # ------------------------------------------------------------------
    # Multi-embodiment routing. IDs must match the action-head projector table.
    # ------------------------------------------------------------------
    action_loss_embodiment_ids: list[int] = field(
        default_factory=lambda: [26, 17, 32]
    )
    max_num_embodiments: int = 32

    # ------------------------------------------------------------------
    # Fine-tune toggles
    # ------------------------------------------------------------------
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    freeze_decode_layer: bool = False
    # Re-anchor global RNG after the first batch fetch and before the first
    # stochastic model forward. This is model-local and does not affect other
    # trainer users.
    train_rng_reset_after_first_batch: bool = True
    train_rng_seed: int | None = None  # None means use training args.seed.
    train_rng_cpu_burn: int = 2

    # ------------------------------------------------------------------
    # Megatron compatibility fields (back-filled in __post_init__)
    # ------------------------------------------------------------------
    num_layers: int = 40
    hidden_size: int = 5120
    num_attention_heads: int = 40
    ffn_hidden_size: int = 13824
    kv_channels: int | None = None
    seq_length: int = 4096
    max_position_embeddings: int = 4096
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    norm_epsilon: float = 1e-6
    # DreamZero checkpoints include q/k/v/o and FFN bias tensors.
    add_bias_linear: bool = True
    add_qkv_bias: bool = True
    untie_embeddings_and_output_weights: bool = True
    add_position_embedding: bool = False
    qk_layernorm: bool = True  # DZ uses RMSNorm on Q/K
    swiglu: bool = False

    # Parallelism placeholders; resolved at trainer entrypoint
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    virtual_pipeline_model_parallel_size: int | None = None
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 1

    # Generic training compatibility knobs
    deallocate_pipeline_outputs: bool = False
    random_fallback_cpu: bool = False
    fp8: bool = False
    fp4: bool = False
    bf16: bool = True
    fp16: bool = False
    enable_autocast: bool = False
    calculate_per_token_loss: bool = False
    init_model_with_meta_device: bool = False
    barrier_with_L1_time: bool = False
    fine_grained_activation_offloading: bool = False
    overlap_moe_expert_parallel_comm: bool = False

    # Synchronisation hook (Megatron expects callable attrs to exist)
    no_sync_func: Any = None

    # Runtime feature mappings populated by Dataset metadata.
    input_features: dict = field(default_factory=dict)
    output_features: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Validation + derived field backfill
    # ------------------------------------------------------------------
    def __post_init__(self):
        """Validate DreamZero settings and populate derived compatibility fields."""
        post_init = getattr(super(), "__post_init__", None)
        if callable(post_init):
            post_init()

        if self.backbone_variant not in _BACKBONE_PRESETS:
            raise ValueError(
                f"backbone_variant must be one of {list(_BACKBONE_PRESETS)}; "
                f"got {self.backbone_variant!r}"
            )
        preset = _BACKBONE_PRESETS[self.backbone_variant]

        # Backbone hyperparams override Megatron compat fields if user did not
        # explicitly tune them. We respect explicit user overrides (i.e. values
        # already different from class defaults are kept as-is).
        if self.num_layers == 40 and preset["num_layers"] != self.num_layers:
            self.num_layers = preset["num_layers"]
        if self.num_attention_heads == 40 and preset["num_heads"] != self.num_attention_heads:
            self.num_attention_heads = preset["num_heads"]
        if self.ffn_hidden_size == 13824 and preset["ffn_dim"] != self.ffn_hidden_size:
            self.ffn_hidden_size = preset["ffn_dim"]

        # Megatron/DiT hidden size and action-head backbone embedding dim.
        self.hidden_size = preset["dim"]
        self.backbone_hidden_size = preset["dim"]
        self.backbone_in_dim = preset["in_dim"]
        self.backbone_out_dim = preset["out_dim"]
        self.backbone_freq_dim = preset["freq_dim"]
        self.backbone_model_type = preset["model_type"]
        self.backbone_frame_seqlen = preset["frame_seqlen"]
        self.backbone_concat_first_frame_latent = preset["concat_first_frame_latent"]
        self.vae_class = preset["vae_class"]
        self.vae_z_dim = preset["vae_z_dim"]
        self.vae_dim = preset["vae_dim"]
        self.target_video_height = preset["target_video_height"]
        self.target_video_width = preset["target_video_width"]

        if self.kv_channels is None and self.num_attention_heads:
            self.kv_channels = max(1, self.backbone_hidden_size // self.num_attention_heads)

        from .precomputed_cache import (
            apply_precomputed_cache_config,
        )

        apply_precomputed_cache_config(self)

        if self.fsdp_initial_param_rounding_dtype is not None:
            rounding_dtype = str(self.fsdp_initial_param_rounding_dtype).strip().lower()
            if rounding_dtype in {"", "0", "false", "no", "off", "none", "null"}:
                self.fsdp_initial_param_rounding_dtype = None
            elif rounding_dtype in {"bf16", "bfloat16"}:
                self.fsdp_initial_param_rounding_dtype = "bf16"
            elif rounding_dtype in {"fp16", "float16", "half"}:
                self.fsdp_initial_param_rounding_dtype = "fp16"
            else:
                raise ValueError(
                    "fsdp_initial_param_rounding_dtype must be one of "
                    "null/bf16/bfloat16/fp16/float16"
                )
        if self.n_action_steps > self.action_horizon:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot exceed "
                f"action_horizon ({self.action_horizon})"
            )

    # ------------------------------------------------------------------
    # PreTrainedConfig hooks
    # ------------------------------------------------------------------
    def validate_features(self) -> None:
        """Ensure input/output feature schema is well-formed."""
        if "observation.state" not in self.input_features:
            self.input_features["observation.state"] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),
            )
        if "action" not in self.output_features:
            self.output_features["action"] = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),
            )

    def get_optimizer_preset(self):
        """Not supported; DreamZero reads optimizer settings from CLI args."""
        raise NotImplementedError(
            "DreamZero optimizer settings are launcher CLI args in the "
            "LoongForge embodied trainer; do not read them from model YAML."
        )

    def get_scheduler_preset(self):
        """Not supported; DreamZero reads scheduler settings from CLI args."""
        raise NotImplementedError(
            "DreamZero scheduler settings are launcher CLI args in the "
            "LoongForge embodied trainer; do not read them from model YAML."
        )

    @property
    def observation_delta_indices(self) -> None:
        """DreamZero does not use observation delta indices."""
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        """Return the list of action-horizon indices."""
        return list(range(self.action_horizon))

    @property
    def reward_delta_indices(self) -> None:
        """DreamZero does not use reward delta indices."""
        return None


def _dreamzero_bool(value: Any) -> bool:
    """Normalize a DreamZero performance option to ``bool``."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _dreamzero_int(value: Any) -> int:
    """Normalize a DreamZero performance option to ``int``."""
    return int(value)


def _dreamzero_str(value: Any) -> str:
    """Normalize a nullable DreamZero performance option to ``str``."""
    return "" if value is None else str(value)


def dreamzero_dit_performance_options(
    model_config: DreamZeroConfig,
) -> dict[str, object]:
    """Collect normalized DiT performance-switch options from ``model_config``."""
    return {
        "compile_causal_attention": _dreamzero_bool(
            model_config.compile_causal_attention
        ),
        "compile_causal_attention_parts": _dreamzero_str(
            model_config.compile_causal_attention_parts
        ),
        "compile_causal_attention_warmup_frames": _dreamzero_str(
            model_config.compile_causal_attention_warmup_frames
        ),
        "compile_causal_attention_warmup_blocks": _dreamzero_str(
            model_config.compile_causal_attention_warmup_blocks
        ),
        "compile_causal_attention_warmup_batch": _dreamzero_int(
            model_config.compile_causal_attention_warmup_batch
        ),
        "compile_causal_attention_warmup_backward": _dreamzero_bool(
            model_config.compile_causal_attention_warmup_backward
        ),
        "compile_causal_cross_attention": _dreamzero_bool(
            model_config.compile_causal_cross_attention
        ),
        "compile_cross_attention_emulate_precision_casts": _dreamzero_bool(
            model_config.compile_cross_attention_emulate_precision_casts
        ),
        "compile_block_norm_modulate": _dreamzero_bool(
            model_config.compile_block_norm_modulate
        ),
        "block_norm_modulate_impl": _dreamzero_str(
            model_config.block_norm_modulate_impl
        ),
        "block_norm_modulate_triton_warps": _dreamzero_int(
            model_config.block_norm_modulate_triton_warps
        ),
        "qk_rmsnorm_impl": _dreamzero_str(
            model_config.qk_rmsnorm_impl
        ),
        "avoid_rope_reconcat": _dreamzero_bool(
            model_config.avoid_rope_reconcat
        ),
        "skip_single_state_attention": _dreamzero_bool(
            model_config.skip_single_state_attention
        ),
        "manual_self_attn_linear_backward": _dreamzero_bool(
            model_config.manual_self_attn_linear_backward
        ),
        "flash_attention_dense": _dreamzero_bool(
            model_config.flash_attention_dense
        ),
        "flash_attention_dense_min_q": _dreamzero_int(
            model_config.flash_attention_dense_min_q
        ),
        "flash_attention_dense_policy": _dreamzero_str(
            model_config.flash_attention_dense_policy
        ),
        "cache_fa_lens": _dreamzero_bool(
            model_config.cache_fa_lens
        ),
        "cache_fa_lens_clone": _dreamzero_bool(
            model_config.cache_fa_lens_clone
        ),
        "fused_rope": _dreamzero_bool(model_config.fused_rope),
        "fused_rope_fp64": _dreamzero_bool(model_config.fused_rope_fp64),
    }
