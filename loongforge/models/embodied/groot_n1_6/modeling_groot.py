"""Model implementation for the Gr00tN1d6 policy.

Copyright 2024 NVIDIA. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Tuple
import logging
import torch
from torch import nn
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature

from .configuration_groot import Gr00tN1d6OmniConfig
from .modules.dit import AlternateVLDiT, DiT
from .eagle3_model import EagleBackbone
from .modules.embodiment_mlp import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)
import warnings
warnings.filterwarnings("ignore", message="torch.get_autocast_gpu_dtype", category=DeprecationWarning)


class Gr00tN1d6ActionHead(nn.Module):
    """Action head component for flow matching diffusion policy."""

    supports_gradient_checkpointing = True

    def __init__(self, config: Gr00tN1d6OmniConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        self.action_dim = config.max_action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps
        
        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # Initialize diffusion model
        if config.use_alternate_vl_dit:
            self.model = AlternateVLDiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
                attend_text_every_n_blocks=config.attend_text_every_n_blocks,
            )
            print("Using AlternateVLDiT for diffusion model")
        else:
            self.model = DiT(
                **config.diffusion_model_cfg, cross_attention_dim=config.backbone_embedding_dim
            )
            print("Using DiT for diffusion model")

        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        # State dropout parameters
        self.state_dropout_prob = config.state_dropout_prob
        self.mask_token = (
            nn.Parameter(0.02 * torch.randn(1, 1, self.input_embedding_dim))
            if self.state_dropout_prob > 0
            else None
        )

        # State noise parameters
        self.state_additive_noise_scale = config.state_additive_noise_scale

        self.beta_dist = torch.distributions.Beta(
            config.noise_beta_alpha, config.noise_beta_beta
        )
        self.num_timestep_buckets = config.num_timestep_buckets
        self.set_trainable_parameters(
            config.tune_projector, config.tune_diffusion_model, config.tune_vlln
        )

    def set_trainable_parameters(
        self, tune_projector: bool, tune_diffusion_model: bool, tune_vlln: bool
    ):
        """
        Set trainable parameters based on configuration flags.

        Args:
            tune_projector: Whether to tune the projector modules
            tune_diffusion_model: Whether to tune the diffusion model
            tune_vlln: Whether to tune the vlln module
        """
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        self.tune_vlln = tune_vlln
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
            if self.state_dropout_prob > 0:
                self.mask_token.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        if not tune_vlln:
            self.vlln.requires_grad_(False)
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        print(f"Tune action head vlln: {self.tune_vlln}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_projector and not tune_diffusion_model and not tune_vlln:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()

    def sample_time(self, batch_size, device, dtype):
        """
        Sample time steps from beta distribution.

        When running inside a CUDA graph capture (full-iteration graph), uses
        the inverse CDF method which is graph-capturable.  For Beta(alpha, 1),
        the inverse CDF is u^(1/alpha), which only requires torch.rand (CUDA RNG).

        For general Beta(alpha, beta) with beta != 1, falls back to
        torch.distributions.Beta.sample() which is NOT graph-capturable and
        will raise a RuntimeError if called during graph capture.

        Args:
            batch_size: Number of samples to generate
            device: Device to place tensors on
            dtype: Data type for tensors

        Returns:
            Sampled time steps
        """
        # Use inverse CDF for Beta(alpha, 1) — always, not just during graph capture.
        # This ensures identical RNG consumption in both eager and CUDA graph modes.
        # For Beta(alpha, 1), the CDF is x^alpha, so the inverse CDF is u^(1/alpha).
        # When --cuda-graph-scope=per_microbatch: use Beta.sample() to achieve
        # bit-exact alignment with pure eager (sample_time runs outside the graph).
        from loongforge.models.common.cuda_graph_config import is_per_microbatch_graph

        if is_per_microbatch_graph():
            sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        elif self.config.noise_beta_beta != 1.0:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "sample_time() during CUDA graph capture requires noise_beta_beta=1.0, "
                    f"got beta={self.config.noise_beta_beta}."
                )
            # Fallback to Beta.sample (beta != 1)
            sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        else:
            u = torch.rand(batch_size, device=device, dtype=dtype)
            sample = u.pow(1.0 / self.config.noise_beta_alpha)
        sample = (1 - sample) * self.config.noise_s
        return sample

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        """
        Process backbone output through vlln module.

        Args:
            backbone_output: BatchFeature containing backbone features

        Returns:
            Processed BatchFeature
        """
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        if isinstance(self.vlln, nn.LayerNorm):
            target_dtype = self.vlln.weight.dtype
            backbone_features = backbone_features.to(dtype=target_dtype)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """
        Forward pass through the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - action: [B, action_horizon, action_dim] (during training)
                - embodiment_id: [B] (embodiment IDs)
                - action_mask: [B, action_horizon, action_dim]

        Returns:
            BatchFeature containing:
                - loss: action prediction loss
        """
        # Set frozen modules to eval mode
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings
        vl_embeds = backbone_output.backbone_features
        device = vl_embeds.device

        # Get state and actions
        state = action_input.state
        actions = action_input.action

        # Get batch size from state (the authoritative source for training batch size)
        state_batch_size = state.shape[0]

        # Ensure actions batch size matches state batch size
        # This handles cases where action processing in modeling file creates mismatched batches
        action_batch_size = actions.shape[0]
        if action_batch_size != state_batch_size:
            if action_batch_size == 1 and state_batch_size > 1:
                # Actions have batch 1 but state has full batch - expand actions
                # This can happen when actions were reshaped incorrectly
                actions = actions.expand(state_batch_size, -1, -1)
                action_batch_size = state_batch_size
            elif state_batch_size == 1 and action_batch_size > 1:
                # Unusual case - state has batch 1, use action batch size
                state_batch_size = action_batch_size

        # Use state batch size as the canonical batch size
        batch_size = state_batch_size

        # Get embodiment ID
        embodiment_id = action_input.embodiment_id

        # Convert to tensor if it's a Python int/float
        if not isinstance(embodiment_id, torch.Tensor):
            embodiment_id = torch.full((batch_size,), embodiment_id, device=device, dtype=torch.long)
        # Ensure embodiment_id is at least 1D [B] for proper indexing
        if embodiment_id.ndim == 0:
            embodiment_id = embodiment_id.unsqueeze(0).expand(batch_size)
        elif embodiment_id.ndim == 1 and embodiment_id.shape[0] != batch_size:
            # Batch size mismatch - expand or truncate to match batch_size
            if embodiment_id.shape[0] == 1:
                embodiment_id = embodiment_id.expand(batch_size)
            else:
                # Use first embodiment ID for all samples (common in single-embodiment training)
                embodiment_id = embodiment_id[:1].expand(batch_size)
        elif embodiment_id.ndim > 1:
            # Flatten if needed (shouldn't happen, but be defensive)
            embodiment_id = embodiment_id.flatten()
            if embodiment_id.shape[0] != batch_size:
                if embodiment_id.shape[0] == 1:
                    embodiment_id = embodiment_id.expand(batch_size)
                else:
                    embodiment_id = embodiment_id[:1].expand(batch_size)

        # Embed state
        # Handle 2D state tensors [B, state_dim] by expanding to 3D [B, 1, state_dim]
        # The state encoder expects 3D input [B, T, state_dim]
        if state.ndim == 2:
            state = state.unsqueeze(1)  # [B, state_dim] -> [B, 1, state_dim]
        state_features = self.state_encoder(state, embodiment_id)

        # Apply state dropout during training
        if self.state_dropout_prob > 0:
            do_dropout = (
                torch.rand(state_features.shape[0], device=state_features.device) < self.state_dropout_prob
            )
            do_dropout = do_dropout[:, None, None].to(dtype=state_features.dtype)
            state_features = state_features * (1 - do_dropout) + self.mask_token * do_dropout

        # Add Gaussian noise to state features during training
        if self.training and self.state_additive_noise_scale > 0:
            noise = torch.randn_like(state_features) * self.state_additive_noise_scale
            state_features = state_features + noise

        # Embed noised action trajectory (flow matching)
        # In per-microbatch graph mode: use external static buffers for noise/time
        # so that graph captures reads from fixed addresses, and we can
        # overwrite them with fresh Beta.sample() values before each replay.
        _noise_buf = getattr(self, '_split_noise_buf', None)
        if _noise_buf is not None:
            # Split graph mode: read from pre-allocated static buffers.
            # Buffers are allocated before capture with correct shape.
            noise = self._split_noise_buf
            t_1d = self._split_time_buf
            t = t_1d[:, None, None]
        else:
            # Record action shape during warmup for later buffer allocation
            if getattr(self, '_split_record_shape', False):
                self._split_actions_shape = actions.shape
                self._split_actions_device = actions.device
                self._split_actions_dtype = actions.dtype
            noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
            t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
            t = t[:, None, None]  # shape (B, 1, 1) for broadcast

        # Interpolate between noise and actions
        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert continuous t to discrete timesteps
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # Add position embedding
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Concatenate state and action embeddings
        sa_embs = torch.cat((state_features, action_features), dim=1)

        # Ensure vl_embeds batch size matches sa_embs batch size
        # The backbone might output batch size 1 if it processes the batch as a single item
        sa_batch_size = sa_embs.shape[0]
        vl_batch_size = vl_embeds.shape[0]
        if vl_batch_size == 1 and sa_batch_size > 1:
            # Expand vl_embeds to match sa_embs batch size
            # Repeat the single batch item for all batches
            vl_embeds = vl_embeds.expand(sa_batch_size, -1, -1)
            # Also expand attention mask if it exists
            if (
                hasattr(backbone_output, "backbone_attention_mask") and
                backbone_output.backbone_attention_mask is not None
            ):
                vl_attn_mask = backbone_output.backbone_attention_mask
                if vl_attn_mask.shape[0] == 1:
                    vl_attn_mask = vl_attn_mask.expand(sa_batch_size, -1)
            else:
                vl_attn_mask = backbone_output.backbone_attention_mask
        else:
            vl_attn_mask = backbone_output.backbone_attention_mask

        # Forward through DiT
        if self.config.use_alternate_vl_dit:
            image_mask = backbone_output.image_mask
            backbone_attention_mask = backbone_output.backbone_attention_mask
            # Expand image_mask and backbone_attention_mask if needed
            if image_mask is not None and image_mask.shape[0] == 1 and sa_batch_size > 1:
                image_mask = image_mask.expand(sa_batch_size, -1)
            if (
                backbone_attention_mask is not None and
                backbone_attention_mask.shape[0] == 1 and
                sa_batch_size > 1
            ):
                backbone_attention_mask = backbone_attention_mask.expand(sa_batch_size, -1)
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
            )
        else:
            # Ensure vl_embeds batch size matches sa_embs batch size (same fix as above)
            sa_batch_size = sa_embs.shape[0]
            vl_batch_size = vl_embeds.shape[0]
            if vl_batch_size == 1 and sa_batch_size > 1:
                vl_embeds = vl_embeds.expand(sa_batch_size, -1, -1)
                if vl_attn_mask is not None and vl_attn_mask.shape[0] == 1:
                    vl_attn_mask = vl_attn_mask.expand(sa_batch_size, -1)
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
            )

        # Decode actions
        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # Compute masked MSE loss
        # Get action_mask from input, or create default (all valid) if missing
        action_mask = getattr(action_input, "action_mask", None)
        if action_mask is None:
            # Create default mask (all valid) matching pred_actions shape
            action_mask = torch.ones_like(pred_actions)
            logging.warning(
                f"action_mask missing in action_input, created default mask with shape {action_mask.shape}"
            )
        else:
            # Expand action_mask to match batch size if needed (fixes batch size mismatch)
            if action_mask.shape[0] != pred_actions.shape[0]:
                # action_mask has batch_size=1 but pred_actions has batch_size=B
                # Expand action_mask: [1, T, D] -> [B, T, D]
                action_mask = action_mask.expand(pred_actions.shape[0], -1, -1)
        # Ensure velocity matches pred_actions shape (in case actions were truncated)
        if velocity.shape[1] != pred_actions.shape[1]:
            velocity = velocity[:, : pred_actions.shape[1], :]
        
        # breakpoint()
        action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = action_loss.sum() / (action_mask.sum() + 1e-6)

        return {
            "loss": loss,
            "action_loss": action_loss,
            "action_mask": action_mask,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }

    def _encode_features(
        self, backbone_output: BatchFeature, action_input: BatchFeature
    ) -> BatchFeature:
        """
        Encode features for the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - state_features: [B, state_horizon, input_embedding_dim]
        """
        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        return BatchFeature(data={"backbone_features": vl_embeds, "state_features": state_features})

    @torch.no_grad()
    def get_action_with_features(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_features: [B, seq_len, backbone_embedding_dim]
            state_features: [B, state_horizon, input_embedding_dim]
            embodiment_id: [B] (embodiment IDs)
            backbone_output: Output from the backbone model
        """
        vl_embeds = backbone_features

        # Set initial actions as the sampled noise.
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.action_dim),
            dtype=vl_embeds.dtype,
            device=device,
        )

        dt = 1.0 / self.num_inference_timesteps

        # Run denoising steps.
        for t in range(self.num_inference_timesteps):
            t_cont = t / float(self.num_inference_timesteps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            # Add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            sa_embs = torch.cat((state_features, action_features), dim=1)

            # Run model forward.
            if self.config.use_alternate_vl_dit:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                )
            else:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                )
            pred = self.action_decoder(model_output, embodiment_id)

            pred_velocity = pred[:, -self.action_horizon :]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity
        return BatchFeature(
            data={
                "action_pred": actions,
                "backbone_features": vl_embeds,
                "state_features": state_features,
            }
        )

    @torch.no_grad()
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - action_pred: [B, action_horizon, action_dim] predicted actions
        """
        features = self._encode_features(backbone_output, action_input)
        return self.get_action_with_features(
            backbone_features=features.backbone_features,
            state_features=features.state_features,
            embodiment_id=action_input.embodiment_id,
            backbone_output=backbone_output,
        )

    @property
    def device(self):
        """Return device of the action-head parameters."""
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        """Return dtype of the model parameters."""
        return next(iter(self.parameters())).dtype

    def prepare_input(self, batch: dict) -> BatchFeature:
        """Prepare input batch for the action head."""
        return BatchFeature(data=batch)


def get_backbone_cls(config: Gr00tN1d6OmniConfig):
    """Get backbone class based on model name in config."""
    if "NVEagle" in config.model_name or "nvidia/Eagle" in config.model_name or "eagle" in config.model_name.lower():
        return EagleBackbone
    else:
        raise ValueError(f"Unsupported model name: {config.model_name}")


class Gr00tN1d6(nn.Module):
    """Gr00tN1d6: Vision-Language-Action model with backbone."""

    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Gr00tN1d6OmniConfig,
        transformers_loading_kwargs: dict | None = None,
    ):
        """
        Initialize Gr00tN1d6 model.

        Args:
            config: Model configuration
            transformers_loading_kwargs: Dict with transformers loading parameters:
                - transformers_trust_remote_code: Whether to trust remote code when loading from HF Hub
                - transformers_local_files_only: Whether to only use local files
                - model_revision: Specific model revision to use
                - transformers_cache_dir: Directory to cache downloaded models
                - transformers_access_token: HuggingFace access token for gated models

        Note: During training, transformers parameters are passed from training config.
              During inference (e.g., from_pretrained), defaults are used.
        """
        super().__init__()
        self.config = config
        if transformers_loading_kwargs is None:
            transformers_loading_kwargs = {"trust_remote_code": True}

        backbone_cls = get_backbone_cls(config)
        self.backbone = backbone_cls(
            model_name=config.model_name,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            select_layer=config.select_layer,
            reproject_vision=config.reproject_vision,
            use_flash_attention=config.use_flash_attention,
            load_bf16=config.load_bf16,
            tune_top_llm_layers=config.tune_top_llm_layers,
            trainable_params_fp32=config.backbone_trainable_params_fp32,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

        # Initialize action head
        self.action_head = Gr00tN1d6ActionHead(config)

        # Detect checkpoint's expected dimensions from loaded weights.
        # The checkpoint may have been trained with different dims than the config.
        self._checkpoint_max_state_dim = self._detect_checkpoint_state_dim()
        self._checkpoint_max_action_dim = self._detect_checkpoint_action_dim()
        self._checkpoint_action_horizon = self._detect_checkpoint_action_horizon()
        # Pre-allocated padding buffers (lazily initialised on first forward call to
        # avoid repeated torch.zeros/torch.full kernel launches inside CUDA Graph).
        self._pad_bufs: dict | None = None
        # Collator is imported here to avoid circular dependency
        try:
            from .processor_groot import Gr00tN1d6DataCollator
            self.collator = Gr00tN1d6DataCollator(
                model_name=config.model_name,
                model_type=config.backbone_model_type,
                tokenizer_assets_repo=getattr(
                    config, "tokenizer_assets_repo", "aravindhs-NV/eagle3-processor-groot-n1d6"
                ),
                transformers_loading_kwargs=transformers_loading_kwargs,
            )
        except ImportError:
            # Processor not available yet (during init), will be set later
            self.collator = None

    # Megatron pipeline APIs expect set_input_tensor even when pipeline parallel size is 1.
    # Provide a no-op shim to satisfy forward_backward_no_pipelining.
    def set_input_tensor(self, input_tensor):
        """Set input tensor for pipeline parallelism."""
        self._input_tensor = input_tensor

    def _detect_checkpoint_state_dim(self) -> int:
        """Detect the checkpoint's expected state dimension from loaded weights.

        Reads the state encoder's first linear weight shape to find the actual
        input_dim the checkpoint was trained with.

        Returns:
            int: The checkpoint's expected state dimension
        """
        state_encoder = self.action_head.state_encoder
        if hasattr(state_encoder, "layer1") and hasattr(state_encoder.layer1, "W"):
            checkpoint_state_dim = int(state_encoder.layer1.W.shape[1])
            if checkpoint_state_dim != self.config.max_state_dim:
                logging.warning(
                    f"Checkpoint expects max_state_dim={checkpoint_state_dim}, "
                    f"but config has max_state_dim={self.config.max_state_dim}. "
                    f"States will be padded/truncated to {checkpoint_state_dim}."
                )
            return checkpoint_state_dim
        # Fallback to config value if detection fails
        return self.config.max_state_dim

    def _detect_checkpoint_action_dim(self) -> int:
        """Detect the checkpoint's expected action dimension from loaded weights.

        Reads the action encoder's W1 weight shape to find the actual
        action_dim the checkpoint was trained with.

        Returns:
            int: The checkpoint's expected action dimension
        """
        action_encoder = self.action_head.action_encoder
        if hasattr(action_encoder, "W1") and hasattr(action_encoder.W1, "W"):
            checkpoint_action_dim = int(action_encoder.W1.W.shape[1])
            if checkpoint_action_dim != self.config.max_action_dim:
                logging.warning(
                    f"Checkpoint expects max_action_dim={checkpoint_action_dim}, "
                    f"but config has max_action_dim={self.config.max_action_dim}. "
                    f"Actions will be padded/truncated to {checkpoint_action_dim}."
                )
            return checkpoint_action_dim
        # Fallback to config value if detection fails
        return self.config.max_action_dim

    def _detect_checkpoint_action_horizon(self) -> int:
        """Detect the checkpoint's expected action horizon from model config.

        The pretrained model may use a diffusion horizon that differs from the
        config's action_horizon. Training actions are padded to the checkpoint
        horizon so the diffusion dynamics remain correct.

        Returns:
            int: The checkpoint's expected action horizon
        """
        checkpoint_horizon = getattr(self.config, "action_horizon", None)
        # action_horizon in Gr00tN1d6OmniConfig is the *model-level* horizon
        # (e.g. 50 for N1.6), so we read it directly.
        if checkpoint_horizon is None:
            return 50  # N1.6 default
        return int(checkpoint_horizon)

    def _init_pad_bufs(self, inputs: dict) -> None:
        """Pre-allocate zero/one-filled buffers for checkpoint-dim padding.

        Called lazily on the first forward pass so that batch size, device, and
        dtypes are all known.  Each buffer covers the *full* expected shape so that
        only a copy_ (no kernel for the zero tail) is needed on every subsequent
        forward pass, eliminating repeated FillFunctor kernel launches.
        """
        bufs: dict = {}
        max_S = self._checkpoint_max_state_dim
        max_D = self._checkpoint_max_action_dim
        exp_T = self._checkpoint_action_horizon

        state = inputs.get("state")
        if state is not None and torch.is_tensor(state):
            dev, dt = state.device, state.dtype
            if state.ndim == 2:
                B = state.shape[0]
                bufs["state_2d"] = torch.zeros(B, max_S, device=dev, dtype=dt)
            elif state.ndim == 3:
                B, T = state.shape[0], state.shape[1]
                bufs["state_3d"] = torch.zeros(B, T, max_S, device=dev, dtype=dt)

        action = inputs.get("action")
        if action is not None and torch.is_tensor(action):
            dev, dt = action.device, action.dtype
            a = action if action.ndim == 3 else action.unsqueeze(1)
            B = a.shape[0]
            bufs["action"] = torch.zeros(B, exp_T, max_D, device=dev, dtype=dt)

        for mask_key in ("action_mask", "action_is_pad"):
            mask = inputs.get(mask_key)
            if mask is None or not torch.is_tensor(mask):
                continue
            dev, dt = mask.device, mask.dtype
            pad_val = 1 if mask_key == "action_is_pad" else 0
            if mask.ndim == 2:
                B = mask.shape[0]
                buf = torch.full((B, exp_T), pad_val, device=dev, dtype=dt)
                bufs[f"{mask_key}_2d"] = buf
            elif mask.ndim == 3:
                B = mask.shape[0]
                buf = torch.full((B, exp_T, max_D), pad_val, device=dev, dtype=dt)
                bufs[f"{mask_key}_3d"] = buf

        self._pad_bufs = bufs

    def _pad_inputs_to_checkpoint_dims(self, inputs: dict) -> dict:
        """Pad / truncate state and action tensors to the dimensions expected by
        the checkpoint weights.

        This mirrors the logic in lerobot's Gr00tN1d6Policy.forward() and ensures
        that batches produced by the preprocessor (which uses the *config* dims,
        e.g. 29) are brought up to the dims the model was actually trained with
        (e.g. 128 for state/action dim, 50 for action horizon).

        Args:
            inputs: Raw input dict from preprocessor (after any renaming).

        Returns:
            A new dict with 'state' and 'action' tensors padded/truncated.
        """
        inputs = dict(inputs)  # shallow copy so we don't mutate the original

        # Lazily initialise pre-allocated padding buffers on the first call
        # (batch size / device / dtype are only known at runtime).
        # Re-initialise if batch size changes (e.g. last micro-batch or eval).
        _first_tensor = next((v for v in inputs.values() if torch.is_tensor(v)), None)
        if _first_tensor is not None:
            _B = _first_tensor.shape[0]
            # Also check that existing buffers match current state ndim to avoid
            # KeyError when state switches between 2D and 3D across iterations.
            _state = inputs.get("state")
            _state_buf_key = None
            if _state is not None and torch.is_tensor(_state) and _state.ndim in (2, 3):
                _state_buf_key = f"state_{_state.ndim}d"
            # Also check mask ndim to avoid KeyError when mask switches between 2D/3D.
            _mask_buf_missing = False
            for _mk in ("action_mask", "action_is_pad"):
                _m = inputs.get(_mk)
                if _m is not None and torch.is_tensor(_m) and _m.ndim in (2, 3):
                    _mk_key = f"{_mk}_{_m.ndim}d"
                    if self._pad_bufs and _mk_key not in self._pad_bufs:
                        _mask_buf_missing = True
                        break
            _need_reinit = (
                self._pad_bufs is None
                or not self._pad_bufs
                or next(iter(self._pad_bufs.values())).shape[0] != _B
                or (_state_buf_key is not None and _state_buf_key not in self._pad_bufs)
                or _mask_buf_missing
            )
            if _need_reinit:
                self._init_pad_bufs(inputs)

        bufs = self._pad_bufs
        max_state_dim = self._checkpoint_max_state_dim
        max_action_dim = self._checkpoint_max_action_dim
        expected_T = self._checkpoint_action_horizon

        # ---- state ----
        state = inputs.get("state")
        if state is not None and torch.is_tensor(state):
            if state.ndim == 2:
                B, D = state.shape
                if D < max_state_dim:
                    buf = bufs["state_2d"]
                    buf.zero_()
                    buf[:, :D].copy_(state)
                    inputs["state"] = buf
                elif D > max_state_dim:
                    inputs["state"] = state[:, :max_state_dim]
            elif state.ndim == 3:
                B, T, D = state.shape
                if D < max_state_dim:
                    buf = bufs["state_3d"]
                    buf.zero_()
                    buf[:, :, :D].copy_(state)
                    inputs["state"] = buf
                elif D > max_state_dim:
                    inputs["state"] = state[:, :, :max_state_dim]

        # ---- action ----
        action = inputs.get("action")
        if action is not None and torch.is_tensor(action):
            # Ensure 3-D: [B, T, D]
            if action.ndim == 2:
                action = action.unsqueeze(1)  # [B, D] -> [B, 1, D]

            B, T, D = action.shape
            need_T_pad = T < expected_T
            need_D_pad = D < max_action_dim

            if need_T_pad or need_D_pad:
                buf = bufs["action"]
                t_copy = min(T, expected_T)
                d_copy = min(D, max_action_dim)
                buf.zero_()
                buf[:, :t_copy, :d_copy].copy_(action[:, :t_copy, :d_copy])
                action = buf
            elif T > expected_T:
                action = action[:, :expected_T, :]
            if D > max_action_dim:
                action = action[:, :, :max_action_dim]

            inputs["action"] = action

        # ---- action_mask / action_is_pad ----
        # action_mask is 3-D [B, T, D]; action_is_pad is 2-D [B, T].
        # Pad tail with 0 (mask) or 1 (is_pad); truncate if too long.
        for mask_key in ("action_mask", "action_is_pad"):
            mask = inputs.get(mask_key)
            if mask is None or not torch.is_tensor(mask):
                continue
            if mask.ndim == 2:
                B, T = mask.shape
                if T < expected_T:
                    buf = bufs[f"{mask_key}_2d"]
                    buf.fill_(1 if mask_key == "action_is_pad" else 0)
                    buf[:, :T].copy_(mask)
                    inputs[mask_key] = buf
                elif T > expected_T:
                    inputs[mask_key] = mask[:, :expected_T]
            elif mask.ndim == 3:
                B, T, D = mask.shape
                need_T_pad = T < expected_T
                need_D_pad = D < max_action_dim
                if need_T_pad or need_D_pad:
                    buf = bufs[f"{mask_key}_3d"]
                    t_copy = min(T, expected_T)
                    d_copy = min(D, max_action_dim)
                    buf.fill_(1 if mask_key == "action_is_pad" else 0)
                    buf[:, :t_copy, :d_copy].copy_(mask[:, :t_copy, :d_copy])
                    mask = buf
                else:
                    if T > expected_T:
                        mask = mask[:, :expected_T, :]
                    if mask.shape[2] > max_action_dim:
                        mask = mask[:, :, :max_action_dim]
                inputs[mask_key] = mask

        return inputs

    def prepare_input(self, inputs: dict) -> Tuple[BatchFeature, BatchFeature]:
        """Prepare inputs for backbone and action head."""

        # NOTE -- currently the eval code doesn't use collator, so we need to add it here
        # this should ideally be fixed upstream
        if "vlm_content" in inputs and self.collator is not None:
            # Fix for n_envs > 1: Process all environments' VLM content, not just the first
            vlm_content_list = inputs["vlm_content"]
            # Ensure vlm_content_list is always a list for consistent processing
            if not isinstance(vlm_content_list, list):
                vlm_content_list = [vlm_content_list]

            # Process all VLM contents through the collator
            prep = self.collator([{"vlm_content": vlm} for vlm in vlm_content_list])["inputs"]
            inputs.pop("vlm_content")
            inputs.update(prep)


        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)


        # Move to device and dtype
        def to_device_with_dtype(x):
            if torch.is_tensor(x):
                if torch.is_floating_point(x):
                    return x.to(self.device, dtype=self.dtype)
                return x.to(self.device)
            if isinstance(x, dict):
                return {k: to_device_with_dtype(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                converted = [to_device_with_dtype(v) for v in x]
                return type(x)(converted)
            return x

        # Simple map for dict inputs
        backbone_inputs_dict = backbone_inputs.data if isinstance(backbone_inputs, BatchFeature) else backbone_inputs
        action_inputs_dict = action_inputs.data if isinstance(action_inputs, BatchFeature) else action_inputs

        backbone_inputs_dict = {k: to_device_with_dtype(v) for k, v in backbone_inputs_dict.items()}
        action_inputs_dict = {k: to_device_with_dtype(v) for k, v in action_inputs_dict.items()}

        backbone_inputs = BatchFeature(data=backbone_inputs_dict)
        action_inputs = BatchFeature(data=action_inputs_dict)

        return backbone_inputs, action_inputs

    def forward(self, inputs: dict) -> BatchFeature:
        """
        Forward pass through the complete model.

        Args:
            inputs: Dictionary containing:
                - Eagle inputs (prefixed with 'eagle_')
                - Action inputs (state, action, embodiment_id, etc.)

        Returns:
            BatchFeature containing loss and other outputs
        """
        # Pad / truncate state and action to the dims the checkpoint expects
        inputs = self._pad_inputs_to_checkpoint_dims(inputs)
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)

        # Use bf16 autocast for forward computation to satisfy FlashAttention
        # requirements while keeping trainable params in fp32 for optimizer precision.
        # This matches lerobot's "bf16 compute, fp32 params" paradigm.
        device_type = torch.device(getattr(self.config, "device", "cuda")).type
        use_bf16 = getattr(self.config, "use_bf16", True)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_bf16):
            backbone_outputs = self.backbone(backbone_inputs)
            action_outputs = self.action_head(backbone_outputs, action_inputs)

        return action_outputs

    def get_action(self, inputs: dict) -> BatchFeature:
        """
        Generate actions using the complete model.
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        device_type = torch.device(getattr(self.config, "device", "cuda")).type
        use_bf16 = getattr(self.config, "use_bf16", True)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_bf16):

            # Forward through backbone
            backbone_outputs = self.backbone(backbone_inputs)
            action_outputs = self.action_head.get_action(backbone_outputs, action_inputs)

        return action_outputs

    @property
    def device(self):
        """Return device of the model parameters."""
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        """Return dtype of the model parameters."""
        return next(iter(self.parameters())).dtype
    
    def state_dict_for_save_checkpoint(self, destination=None, prefix='', keep_vars=False):
        """Return state dict for checkpoint saving, required by Megatron.
        """
        return self.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)