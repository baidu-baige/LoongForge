import warnings
from collections import deque
from typing import Any

import torch

from ....models.vla.giga_brain_0.modeling_giga_brain_0 import GigaBrain07Policy
from ...pipeline import BasePipeline
from .giga_brain_0_utils import (
    AbsoluteActions,
    ActionStateDimLayout,
    Embodiment3QuaternionTo6D,
    ImageTransform,
    Normalize,
    PadStatesAndActions,
    PromptTokenizerTransform,
    Unnormalize,
    infer_end_effector_type_from_delta_mask,
    resolve_action_state_dim_layout,
)


class GigaBrain07Pipeline(BasePipeline):
    """Pipeline wrapper for GigaBrain07 policy inference and utilities.

    This pipeline handles input preprocessing (state normalization, image transformation, tokenization), calls the underlying policy to sample actions
    and optionally 2D trajectories, and postprocesses outputs back to the original scale and absolute action space.
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_model_path: str,
        fast_tokenizer_path: str,
        embodiment_id: int,
        state_norm_stats: dict,
        action_norm_stats: dict,
        delta_mask: list[bool],
        original_action_dim: int,
        discrete_state_input: bool = True,
        encode_sub_task_input: bool = True,
        autoregressive_inference_mode: bool = False,
        depth_img_prefix_name: str | None = None,
        enable_control_mode_token: bool = False,
        control_mode_override: str | None = None,
        enable_end_effector_token: bool = False,
        end_effector_override: str | None = None,
        delta_mask_selector: str = 'embodiment_id',
        robot_type: str | None = None,
        resize_imgs_with_padding: tuple[int, int] | list[int] = (224, 224),
        prompt_max_length: int = 200,
        prefix_lm_text: bool = False,
        fast_token_vocab_mode: str | None = None,
        eef_dual_hand_prefix_to_6d: bool = False,
        policy: GigaBrain07Policy | None = None,
    ):
        """Initialize the GigaBrain07 pipeline.

        Args:
            model_path: Path to the model checkpoint directory.
            tokenizer_model_path: Path to the tokenizer model.
            fast_tokenizer_path: Path to the fast tokenizer model.
            embodiment_id: Embodiment identifier of the robot/task.
            state_norm_stats: Normalization stats for state.
            action_norm_stats: Normalization stats for action.
            delta_mask: Boolean mask indicating which action dimensions are delta-controlled.
            original_action_dim: Expected original action vector dimension.
            discrete_state_input: Whether to use discrete state input.
            encode_sub_task_input: Whether to include an explicit subtask suffix from the task text.
            autoregressive_inference_mode: Whether to use autoregressive inference mode.
            depth_img_prefix_name: Optional prefix for depth image keys when depth is enabled.
            control_mode_override: Optional explicit control mode used during inference prompt construction.
            end_effector_override: Optional explicit end-effector type used during inference prompt construction.
            resize_imgs_with_padding: Target image size (width, height) used by image and prompt transforms.
            prompt_max_length: Maximum tokenized prompt length.
            prefix_lm_text: Whether to use prefix-LM text masking in tokenizer transform.
            fast_token_vocab_mode: Optional checkpoint compatibility override for FAST-token ids.
            eef_dual_hand_prefix_to_6d: Match training ``GigaBrain07Transform(eef_dual_hand_prefix_to_6d=True)``:
                convert only the leading 16-D dual-hand TCP quaternion block to 6D and keep suffix dims.
        """
        super().__init__()
        load_kwargs = {}
        if fast_token_vocab_mode is not None:
            load_kwargs['fast_token_vocab_mode'] = fast_token_vocab_mode
        self.policy = policy if policy is not None else GigaBrain07Policy.from_pretrained(model_path, **load_kwargs)
        self.policy.eval()
        self.policy_triton = None
        self.embodiment_id = embodiment_id
        self._eef_dual_hand_prefix_to_6d = bool(eef_dual_hand_prefix_to_6d)
        if delta_mask_selector not in ('embodiment_id', 'robot_type'):
            raise ValueError(f'Unsupported delta_mask_selector: {delta_mask_selector!r}')
        self.delta_mask_selector = delta_mask_selector
        if delta_mask_selector == 'robot_type' and not robot_type:
            raise ValueError("robot_type is required when delta_mask_selector='robot_type'")
        self.robot_type = robot_type
        self.device = 'cpu'
        inferred_end_effector_type = infer_end_effector_type_from_delta_mask(delta_mask)
        self.end_effector_type = end_effector_override if end_effector_override is not None else inferred_end_effector_type
        self.resize_imgs_with_padding = tuple(int(value) for value in resize_imgs_with_padding)
        if len(self.resize_imgs_with_padding) != 2:
            raise ValueError(f'resize_imgs_with_padding must contain exactly 2 values, got {resize_imgs_with_padding!r}')
        self.observation_memory_size = int(getattr(self.policy.config, 'observation_memory_size', 1))
        if self.observation_memory_size < 1:
            raise ValueError(f'observation_memory_size must be positive, got {self.observation_memory_size}')
        self.state_input_mode = getattr(self.policy.config, 'state_input_mode', 'prompt')
        if self.state_input_mode not in ('prompt', 'proprio_memory', 'proprio_anchor'):
            raise ValueError(f'Unsupported policy state_input_mode: {self.state_input_mode!r}')
        if self.state_input_mode in ('proprio_memory', 'proprio_anchor') and discrete_state_input:
            raise ValueError(f'state_input_mode={self.state_input_mode!r} is mutually exclusive with ' 'discrete_state_input=True')
        self._image_memory: dict[str, deque[torch.Tensor]] = {}
        self._state_memory: deque[torch.Tensor] = deque(maxlen=self.observation_memory_size)

        self.enable_depth_img = self.policy.vision_in_channels == 4

        # Input transforms
        self.state_normalize_transform = Normalize({embodiment_id: state_norm_stats}, use_quantiles=True)
        self.image_transform = ImageTransform(
            is_train=False,
            resize_imgs_with_padding=self.resize_imgs_with_padding,
            enable_image_aug=False,
            enable_depth_img=self.enable_depth_img,
            depth_img_prefix_name=depth_img_prefix_name,
            vlm_type=self.policy.vlm_type,
        )
        self.prompt_tokenizer_transform = PromptTokenizerTransform(
            is_train=False,
            tokenizer_model_path=tokenizer_model_path,
            fast_tokenizer_path=fast_tokenizer_path,
            max_length=prompt_max_length,
            discrete_state_input=discrete_state_input,
            encode_action_input=False,
            fast_token_vocab_mode=getattr(self.policy, 'fast_token_vocab_mode', None),
            fast_token_tail_skip_tokens=getattr(self.policy, 'fast_token_tail_skip_tokens', 128),
            fast_token_tail_vocab_size=getattr(self.policy, 'fast_token_tail_vocab_size', None),
            encode_sub_task_input=encode_sub_task_input,
            control_mode_override=control_mode_override,
            enable_control_mode_token=enable_control_mode_token,
            end_effector_override=self.end_effector_type,
            enable_end_effector_token=enable_end_effector_token,
            autoregressive_inference_mode=autoregressive_inference_mode,
            vlm_type=self.policy.vlm_type,
            prefix_lm_text=prefix_lm_text,
            resize_imgs_with_padding=self.resize_imgs_with_padding,
            state_input_mode=self.state_input_mode,
        )
        if self.state_input_mode == 'proprio_anchor':
            if self.prompt_tokenizer_transform.propri_token_id != self.policy.propri_token_id:
                raise ValueError(
                    'Tokenizer/model <|propri|> token id mismatch: '
                    f'{self.prompt_tokenizer_transform.propri_token_id} vs '
                    f'{self.policy.propri_token_id}'
                )
        self.embodiment3_quaternion_to_6d_transform = Embodiment3QuaternionTo6D()
        self.pad_states_and_actions_transform = PadStatesAndActions(action_dim=self.policy.max_action_dim)
        self.action_output_dim_mask: torch.Tensor | None = None
        # Output transforms
        self.state_unnormalize_transform = Unnormalize({embodiment_id: state_norm_stats}, use_quantiles=True)
        self.action_unnormalize_transform = Unnormalize({embodiment_id: action_norm_stats}, use_quantiles=True)
        raw_delta_mask_tensor = torch.as_tensor(delta_mask, dtype=torch.bool)
        delta_mask_tensor = raw_delta_mask_tensor
        if self._eef_dual_hand_prefix_to_6d:
            delta_mask_tensor = Embodiment3QuaternionTo6D.transform_mask_dual_hand_quat_prefix(delta_mask_tensor)
        elif Embodiment3QuaternionTo6D.supports_embodiment(embodiment_id):
            delta_mask_tensor = Embodiment3QuaternionTo6D.transform_mask(delta_mask_tensor, embodiment_id)
        self.delta_mask = delta_mask_tensor
        if delta_mask_selector == 'robot_type':
            self.absolute_actions_transform = AbsoluteActions({robot_type: delta_mask_tensor}, selector='robot_type')
        else:
            self.absolute_actions_transform = AbsoluteActions({embodiment_id: raw_delta_mask_tensor})
        if self._eef_dual_hand_prefix_to_6d:
            self.original_action_dim = Embodiment3QuaternionTo6D.transform_output_dim_dual_hand_prefix(original_action_dim)
        elif Embodiment3QuaternionTo6D.has_source_pose_dim(original_action_dim, embodiment_id):
            self.original_action_dim = Embodiment3QuaternionTo6D.transform_dim(original_action_dim, embodiment_id)
        else:
            self.original_action_dim = original_action_dim

    def set_action_output_dim_mask(self, dim_mask: torch.Tensor | list[bool] | None) -> None:
        if dim_mask is None:
            self.action_output_dim_mask = None
            return
        mask = torch.as_tensor(dim_mask, dtype=torch.bool)
        if int(mask.numel()) != int(self.policy.max_action_dim):
            aligned = torch.zeros(int(self.policy.max_action_dim), dtype=torch.bool)
            valid_cols = min(int(mask.numel()), int(aligned.numel()))
            if valid_cols > 0:
                aligned[:valid_cols] = mask[:valid_cols]
            mask = aligned
        self.action_output_dim_mask = mask.to(self.device)

    def _mask_action_output_dims(self, action: torch.Tensor, *, action_dim: int | None = None) -> torch.Tensor:
        if self.action_output_dim_mask is None:
            mask = torch.ones(int(action.shape[-1]), dtype=torch.bool, device=action.device)
        else:
            mask = self.action_output_dim_mask.to(device=action.device)
        valid_cols = min(int(mask.numel()), int(action.shape[-1]))
        if valid_cols <= 0:
            return action
        if valid_cols != int(action.shape[-1]):
            aligned = torch.zeros(int(action.shape[-1]), dtype=torch.bool, device=action.device)
            aligned[:valid_cols] = mask[:valid_cols]
            mask = aligned
        else:
            mask = mask[:valid_cols]
        if action_dim is not None and int(action_dim) < int(action.shape[-1]):
            mask = mask.clone()
            mask[max(0, int(action_dim)) :] = False
        view_shape = [1] * action.ndim
        view_shape[-1] = int(action.shape[-1])
        return torch.where(mask.view(*view_shape), action, torch.zeros((), dtype=action.dtype, device=action.device))

    def _quaternion_to_6d_state(self, state: torch.Tensor) -> torch.Tensor:
        if self._eef_dual_hand_prefix_to_6d:
            return Embodiment3QuaternionTo6D.transform_tensor_dual_hand_quat_prefix(state)
        return self.embodiment3_quaternion_to_6d_transform({'observation.state': state, 'embodiment_id': self.embodiment_id})['observation.state']

    def reset_observation_memory(self) -> None:
        self._image_memory.clear()
        self._state_memory.clear()

    def _stack_observation_memory(self, images: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Stack image history into temporal tensors.

        Returns:
            stacked_images: dict mapping each key to a (T,C,H,W) tensor.
            pad_masks: dict mapping each '{key}_is_pad' to a (T,) bool tensor
                where True means the frame is a cold-start repeat (padded).
                Empty when observation_memory_size == 1 or caller pre-stacked.
        """
        pad_masks: dict[str, torch.Tensor] = {}
        if self.observation_memory_size == 1:
            return images, pad_masks
        has_temporal_image = any(image.ndim == 4 for image in images.values())
        if has_temporal_image:
            for key, image in images.items():
                if image.ndim != 4:
                    raise ValueError(f'When passing image history directly, all images must have shape (T,C,H,W); {key!r} has {tuple(image.shape)}')
                if image.shape[0] != self.observation_memory_size:
                    raise ValueError(
                        f'Image {key!r} has {image.shape[0]} frames but observation_memory_size={self.observation_memory_size}; '
                        f'ensure delta_info offsets count matches observation_memory_size in the model config'
                    )
            return images, pad_masks

        stacked_images: dict[str, torch.Tensor] = {}
        for key, image in images.items():
            if image.ndim != 3:
                raise ValueError(f'Expected current-frame image {key!r} to have shape (C,H,W), got {tuple(image.shape)}')
            memory = self._image_memory.setdefault(key, deque(maxlen=self.observation_memory_size))
            memory.append(image.detach().clone())
            num_real = len(memory)
            num_pad = self.observation_memory_size - num_real
            frames = list(memory)
            if num_pad > 0:
                frames = [frames[0]] * num_pad + frames
            stacked_images[key] = torch.stack(frames, dim=0)
            is_pad = torch.zeros(self.observation_memory_size, dtype=torch.bool, device=image.device)
            is_pad[:num_pad] = True
            pad_masks[f'{key}_is_pad'] = is_pad
        return stacked_images, pad_masks

    def _merge_image_pad_masks(
        self,
        memory_pad_masks: dict[str, torch.Tensor],
        image_pad_masks: dict[str, torch.Tensor] | None,
    ) -> dict[str, torch.Tensor]:
        if image_pad_masks is None:
            return memory_pad_masks

        merged = dict(memory_pad_masks)
        for key, mask in image_pad_masks.items():
            pad_key = key if key.endswith('_is_pad') else f'{key}_is_pad'
            merged[pad_key] = torch.as_tensor(mask, dtype=torch.bool).to(self.device)
        return merged

    def _stack_state_memory(
        self,
        state: torch.Tensor,
        state_memory_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.state_input_mode != 'proprio_memory':
            return None, None

        if state.ndim == 2:
            if state.shape[0] != self.observation_memory_size:
                raise ValueError(
                    f'Pre-stacked state memory has {state.shape[0]} frames but ' f'observation_memory_size={self.observation_memory_size}'
                )
            if state_memory_mask is None:
                state_memory_mask = torch.ones(
                    self.observation_memory_size,
                    dtype=torch.bool,
                    device=state.device,
                )
            else:
                state_memory_mask = torch.as_tensor(state_memory_mask, dtype=torch.bool, device=state.device)
                if state_memory_mask.shape != (self.observation_memory_size,):
                    raise ValueError(f'state_memory_mask must have shape ({self.observation_memory_size},), ' f'got {tuple(state_memory_mask.shape)}')
            return state, state_memory_mask

        if state.ndim != 1:
            raise ValueError(f'state must have shape [D] or [K,D], got {tuple(state.shape)}')

        self._state_memory.append(state.detach().clone())
        num_real = len(self._state_memory)
        num_pad = self.observation_memory_size - num_real
        frames = list(self._state_memory)
        if num_pad > 0:
            frames = [frames[0]] * num_pad + frames
        stacked = torch.stack(frames, dim=0)
        mask = torch.ones(self.observation_memory_size, dtype=torch.bool, device=state.device)
        if num_pad > 0:
            mask[:num_pad] = False
        return stacked, mask

    def _prepare_proprio_anchor(
        self,
        state: torch.Tensor,
        agent_pos_mask: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.state_input_mode != 'proprio_anchor':
            return None, None
        current_state = state[-1] if state.ndim == 2 else state
        current_mask = agent_pos_mask[-1] if agent_pos_mask.ndim == 2 else agent_pos_mask
        if current_state.ndim != 1:
            raise ValueError(f'proprio_anchor expects state [D] or [K,D], got {tuple(state.shape)}')
        if current_state.shape[-1] > self.policy.propri_dim:
            raise ValueError(f'state width {current_state.shape[-1]} exceeds configured proprio dim ' f'{self.policy.propri_dim}')
        pad_width = self.policy.propri_dim - current_state.shape[-1]
        if pad_width > 0:
            current_state = torch.nn.functional.pad(current_state, (0, pad_width))
            current_mask = torch.nn.functional.pad(current_mask, (0, pad_width))
        return current_state[None, None, :], current_mask[None, None, :]

    def to(self, device: str | torch.device):
        self.device = device
        if self.policy_triton is not None:
            self.policy_triton.to(device)
        else:
            self.policy.to(device)
        self.state_normalize_transform.to(device)
        if self.action_output_dim_mask is not None:
            self.action_output_dim_mask = self.action_output_dim_mask.to(device)
        self.state_unnormalize_transform.to(device)
        self.action_unnormalize_transform.to(device)
        self.absolute_actions_transform.to(device)
        self.prompt_tokenizer_transform.to(device)
        return self

    def _resolve_action_state_layout(
        self,
        state: torch.Tensor,
        *,
        is_robot_moving: bool = False,
        is_body_moving: bool = False,
    ) -> ActionStateDimLayout:
        return resolve_action_state_dim_layout(
            self.delta_mask,
            raw_action_dim=self.original_action_dim,
            raw_state_dim=int(state.shape[-1]),
            max_action_dim=int(self.policy.max_action_dim),
            is_robot_moving=is_robot_moving,
            is_body_moving=is_body_moving,
        )

    @staticmethod
    def _mask_state_inputs(
        state: torch.Tensor,
        agent_pos_mask: torch.Tensor | None,
        *,
        state_input_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if state_input_dim < int(state.shape[-1]):
            state = state.clone()
            state[..., state_input_dim:] = 0
        if agent_pos_mask is not None and state_input_dim < int(agent_pos_mask.shape[-1]):
            agent_pos_mask = agent_pos_mask.clone()
            agent_pos_mask[..., state_input_dim:] = 0
        return state, agent_pos_mask

    def _mask_state_for_prompt(
        self,
        state: torch.Tensor,
        *,
        is_robot_moving: bool = False,
        is_body_moving: bool = False,
    ) -> torch.Tensor:
        layout = self._resolve_action_state_layout(
            state,
            is_robot_moving=is_robot_moving,
            is_body_moving=is_body_moving,
        )
        return self._mask_state_inputs(state, None, state_input_dim=layout.state_input_dim)[0]

    def quantize(self) -> None:
        """Apply dynamic float8 quantization to the Paligemma blocks only."""
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, quantize_

        # Only quantize the paligemma part. Skip the action expert part.
        layers = self.policy.paligemma_with_expert.layers
        for i in range(len(layers)):
            quantize_(layers[i].mlps[0], Float8DynamicActivationFloat8WeightConfig())
            quantize_(layers[i].self_attn.q_proj[0], Float8DynamicActivationFloat8WeightConfig())
            quantize_(layers[i].self_attn.k_proj[0], Float8DynamicActivationFloat8WeightConfig())
            quantize_(layers[i].self_attn.v_proj[0], Float8DynamicActivationFloat8WeightConfig())
            quantize_(layers[i].self_attn.o_proj[0], Float8DynamicActivationFloat8WeightConfig())

    def compile(self, backend='inductor', **kwargs: Any) -> None:
        """Compile the `sample_actions` method using `torch.compile` for
        improved runtime speed."""
        if backend == 'triton':
            if self.state_input_mode in ('proprio_memory', 'proprio_anchor'):
                warnings.warn(
                    'Triton backend has a fixed PaliGemma2 encoder layout and does not '
                    f'kernelize {self.state_input_mode} state tokens; falling back to torch.compile.',
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.policy.sample_actions = torch.compile(self.policy.sample_actions, **kwargs)
                if self.policy_triton is not None:
                    self.policy_triton.to('cpu')
                    torch.cuda.empty_cache()
                    self.policy_triton = None
                return

            from ....models.vla.giga_brain_0.triton import GigaBrain07Triton

            assert self.policy.vlm_type == 'paligemma2', 'Only PaliGemma2 is supported for Triton backend.'
            assert not self.policy.enable_learnable_traj_token, 'Trajectory token is not supported for Triton backend.'

            self.policy.to('cpu')
            torch.cuda.empty_cache()
            self.policy_triton = GigaBrain07Triton.from_torch_model(self.policy, device=self.device, emb_id=self.embodiment_id)
        else:
            self.policy.sample_actions = torch.compile(self.policy.sample_actions, **kwargs)
            if self.policy_triton is not None:
                self.policy_triton.to('cpu')
                torch.cuda.empty_cache()
                self.policy_triton = None

    @torch.no_grad()
    def __call__(
        self,
        images: dict[str, torch.Tensor],
        task: str,
        state: torch.Tensor,
        enable_2d_traj_output: bool = False,
        autoregressive_mode_only: bool = False,
        control_mode_override: str | None = None,
        end_effector_override: str | None = None,
        image_pad_masks: dict[str, torch.Tensor] | None = None,
        state_memory_mask: torch.Tensor | None = None,
        is_robot_moving: bool = False,
        is_body_moving: bool = False,
        return_normalized_action: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run policy inference to get the predicted action and optional 2D
        trajectory.

        Args:
            images: Observation images keyed by camera name.
            task: The executing task description.
            state: The joint state tensor.
            enable_2d_traj_output: If True, also return predicted 2D trajectory.
            autoregressive_mode_only: If True, only use autoregressive mode to predict actions.
            control_mode_override: Optional explicit control mode used during prompt construction.
            end_effector_override: Optional explicit end-effector type used during prompt construction.
            image_pad_masks: Optional temporal image padding masks keyed by
                ``'{image_key}_is_pad'``. This is used when callers pass
                pre-stacked memory images directly.
            state_memory_mask: Optional valid-frame mask for pre-stacked state
                memory. True means the state frame is valid.
            is_robot_moving: Whether action-only mobile-base dimensions are active.
            is_body_moving: Whether the full state-aligned body action prefix is
                active when the mobile base is stationary. This does not change
                the state input dimensions.
            return_normalized_action: If True, also return the raw sampled
                normalized-delta action before unnormalization and absolute-action
                conversion.

        Returns:
            If `enable_2d_traj_output` is False, returns the predicted action tensor.
            Otherwise, returns a tuple of (predicted action tensor, 2D trajectory tensor).
        """
        if autoregressive_mode_only:
            return self.predict_autoregressive_actions(
                images,
                task,
                state,
                control_mode_override=control_mode_override,
                end_effector_override=end_effector_override,
                image_pad_masks=image_pad_masks,
                state_memory_mask=state_memory_mask,
                is_robot_moving=is_robot_moving,
                is_body_moving=is_body_moving,
            )

        # Input transforms
        ori_device = state.device
        state = state.to(self.device)
        for key in images:
            images[key] = images[key].to(self.device)
        images, memory_pad_masks = self._stack_observation_memory(images)
        memory_pad_masks = self._merge_image_pad_masks(memory_pad_masks, image_pad_masks)

        images, img_masks, image_transform_params = self.image_transform({**images, **memory_pad_masks})
        state = self._quaternion_to_6d_state(state)
        agent_pos_mask = (~torch.isnan(state)).to(dtype=torch.float32)
        if self.state_input_mode == 'proprio_anchor':
            state = torch.nan_to_num(state, nan=0.0)
        state = self.state_normalize_transform(state, embodiment_id=self.embodiment_id)
        if self.state_input_mode == 'proprio_anchor':
            state = state.clamp(-1.0, 1.0)
        action_reference_state = state
        action_state_layout = self._resolve_action_state_layout(
            state,
            is_robot_moving=is_robot_moving,
            is_body_moving=is_body_moving,
        )
        state, agent_pos_mask = self._mask_state_inputs(
            state,
            agent_pos_mask,
            state_input_dim=action_state_layout.state_input_dim,
        )
        current_state = state[-1] if state.ndim == 2 else state
        state_memory, state_memory_masks = self._stack_state_memory(state, state_memory_mask)
        proprioception, anchor_agent_pos_mask = self._prepare_proprio_anchor(state, agent_pos_mask)
        prompt_inputs = {'task': task, **image_transform_params}
        if self.state_input_mode == 'prompt':
            prompt_state = self._mask_state_for_prompt(
                current_state,
                is_robot_moving=is_robot_moving,
                is_body_moving=is_body_moving,
            )
            prompt_inputs['observation.state'] = prompt_state
        if control_mode_override is not None:
            prompt_inputs['control_mode_override'] = control_mode_override
        if end_effector_override is not None:
            prompt_inputs['end_effector_override'] = end_effector_override
        lang_tokens, lang_masks, lang_att_masks, _, _, _, _ = self.prompt_tokenizer_transform(prompt_inputs)
        state = self.pad_states_and_actions_transform({'observation.state': current_state})['observation.state']
        emb_ids = torch.tensor(self.embodiment_id, dtype=torch.long, device=self.device)
        emb_ids = emb_ids[None, ...]
        for i in range(len(images)):
            images[i] = images[i][None, ...]
            img_masks[i] = img_masks[i][None, ...]
        lang_tokens = lang_tokens[None, ...]
        lang_masks = lang_masks[None, ...]
        lang_att_masks = lang_att_masks[None, ...]

        # Inference
        model = self.policy_triton if self.policy_triton is not None else self.policy.sample_actions
        outputs = model(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            emb_ids,
            enable_2d_traj_output=enable_2d_traj_output,
            lang_att_masks=lang_att_masks,
            state_memory=state_memory[None, ...] if state_memory is not None else None,
            state_memory_masks=state_memory_masks[None, ...] if state_memory_masks is not None else None,
            proprioception=proprioception,
            agent_pos_mask=anchor_agent_pos_mask,
        )
        if enable_2d_traj_output:
            pred_action, traj_pred = outputs
        else:
            pred_action = outputs

        pred_action = self._mask_action_output_dims(pred_action, action_dim=action_state_layout.action_supervised_dim)
        pred_normalized_action = pred_action[0].to(ori_device) if return_normalized_action else None

        # Output transforms
        output_dict = {
            'action': pred_action[0],
            'observation.state': action_reference_state,
            'embodiment_id': self.embodiment_id,
            'robot_type': self.robot_type,
        }
        output_dict['observation.state'] = self.state_unnormalize_transform(output_dict['observation.state'], embodiment_id=self.embodiment_id)
        output_dict['action'] = self.action_unnormalize_transform(output_dict['action'], embodiment_id=self.embodiment_id)
        output_dict = self.absolute_actions_transform(output_dict)
        pred_action = output_dict['action'][:, : self.original_action_dim].to(ori_device)
        pred_action = self._mask_action_output_dims(pred_action, action_dim=action_state_layout.action_supervised_dim)
        if enable_2d_traj_output:
            traj_pred = traj_pred[0]
            if 'resize_with_pad' in image_transform_params:
                ratio = image_transform_params['resize_with_pad']['ratio']
                pad_x, pad_y = image_transform_params['resize_with_pad']['padding']
                traj_pred[:, ::2] = (traj_pred[:, ::2] * self.resize_imgs_with_padding[0] - pad_x) * ratio
                traj_pred[:, 1::2] = (traj_pred[:, 1::2] * self.resize_imgs_with_padding[1] - pad_y) * ratio
            traj_pred = traj_pred.to(ori_device)

            if return_normalized_action:
                return pred_action, traj_pred, pred_normalized_action
            return pred_action, traj_pred

        if return_normalized_action:
            return pred_action, pred_normalized_action
        return pred_action

    def _tokenize_autoregressive_prompt(self, prompt_text: str) -> tuple[torch.Tensor, torch.Tensor]:
        tokenizer = self.prompt_tokenizer_transform.paligemma_tokenizer
        max_length = self.prompt_tokenizer_transform.max_length

        prompt_output = tokenizer(prompt_text, add_special_tokens=True, return_tensors='pt', truncation=False)
        prompt_ids = prompt_output['input_ids'].squeeze(0)
        prompt_mask = prompt_output['attention_mask'].squeeze(0)

        if prompt_ids.shape[0] >= max_length:
            raise ValueError(f'Prompt length {prompt_ids.shape[0]} exceeds max length {max_length}')

        padded_output = tokenizer.pad(
            {
                'input_ids': prompt_ids.tolist(),
                'attention_mask': prompt_mask.tolist(),
            },
            padding='max_length',
            padding_side='left',
            max_length=max_length,
            return_tensors='pt',
        )

        lang_tokens = padded_output['input_ids'].squeeze(0).to(dtype=torch.int32, device=self.device)
        lang_masks = padded_output['attention_mask'].squeeze(0).to(dtype=torch.bool, device=self.device)
        return lang_tokens, lang_masks

    @torch.no_grad()
    def predict_current_subtask(
        self,
        images: dict[str, torch.Tensor],
        task: str,
        state: torch.Tensor | None = None,
        control_mode_override: str | None = None,
        end_effector_override: str | None = None,
        max_new_tokens: int = 64,
        state_is_normalized: bool = False,
        is_robot_moving: bool = False,
        is_body_moving: bool = False,
    ) -> list[str]:
        """Predict the current subtask from images and task description.

        Args:
            images: Observation images keyed by camera name.
            task: The executing task description.
            state: Optional state tensor for discrete state prompt tokens.
            state_is_normalized: Whether ``state`` has already gone through
                quaternion conversion and normalization.
            is_robot_moving: Whether action-only mobile-base dimensions are active.
            is_body_moving: Body-action supervision/output selector. This does
                not change the state input dimensions.

        Returns:
            List of predicted subtask strings, one per sample in the batch.
        """
        tokenizer = self.prompt_tokenizer_transform.paligemma_tokenizer

        if state is not None:
            state = state.to(self.device)
        for key in images:
            images[key] = images[key].to(self.device)

        images, img_masks, image_transform_params = self.image_transform(images)
        prompt_inputs = {'task': task, 'embodiment_id': self.embodiment_id, **image_transform_params}
        proprioception = None
        anchor_agent_pos_mask = None
        if state is not None:
            if not state_is_normalized:
                state = self._quaternion_to_6d_state(state)
                agent_pos_mask = (~torch.isnan(state)).to(dtype=torch.float32)
                if self.state_input_mode == 'proprio_anchor':
                    state = torch.nan_to_num(state, nan=0.0)
                state = self.state_normalize_transform(state, embodiment_id=self.embodiment_id)
            else:
                agent_pos_mask = (~torch.isnan(state)).to(dtype=torch.float32)
                if self.state_input_mode == 'proprio_anchor':
                    state = torch.nan_to_num(state, nan=0.0)
            if self.state_input_mode == 'proprio_anchor':
                state = state.clamp(-1.0, 1.0)
            action_state_layout = self._resolve_action_state_layout(
                state,
                is_robot_moving=is_robot_moving,
                is_body_moving=is_body_moving,
            )
            state, agent_pos_mask = self._mask_state_inputs(
                state,
                agent_pos_mask,
                state_input_dim=action_state_layout.state_input_dim,
            )
            if self.state_input_mode == 'proprio_anchor':
                proprioception, anchor_agent_pos_mask = self._prepare_proprio_anchor(state, agent_pos_mask)
            else:
                state = self._mask_state_for_prompt(
                    state,
                    is_robot_moving=is_robot_moving,
                    is_body_moving=is_body_moving,
                )
                prompt_inputs['observation.state'] = state
        elif self.state_input_mode == 'proprio_anchor':
            raise ValueError('predict_current_subtask requires state in proprio_anchor mode')
        if control_mode_override is not None:
            prompt_inputs['control_mode_override'] = control_mode_override
        if end_effector_override is not None:
            prompt_inputs['end_effector_override'] = end_effector_override
        lang_tokens, lang_masks, lang_att_masks, _, _, _, _ = self.prompt_tokenizer_transform.create_subtask_inference_tokens(prompt_inputs)

        generated = self.generate_autoregressive_tokens(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            lang_att_masks=lang_att_masks,
            max_new_tokens=max_new_tokens,
            proprioception=proprioception,
            agent_pos_mask=anchor_agent_pos_mask,
        )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        return decoded

    @torch.no_grad()
    def predict_vqa(
        self,
        images: dict[str, torch.Tensor],
        question: str,
        prompt_text: str | None = None,
        max_new_tokens: int = 64,
    ) -> list[str]:
        """Predict VQA answers from images and question text.

        Args:
            images: Observation images keyed by camera name.
            question: The question to answer.
            prompt_text: Optional full prompt text. Defaults to a standard
                Question/Answer template when not provided.
            max_new_tokens: Maximum number of generated answer tokens.

        Returns:
            List of decoded answer strings, one per sample in the batch.
        """
        tokenizer = self.prompt_tokenizer_transform.paligemma_tokenizer
        if self.state_input_mode == 'proprio_anchor':
            raise ValueError('predict_vqa does not provide state required by proprio_anchor mode')

        for key in images:
            images[key] = images[key].to(self.device)

        images, img_masks, _ = self.image_transform(images)
        if prompt_text is None:
            prompt_text = f'Question: {question}\nAnswer:'
        lang_tokens, lang_masks = self._tokenize_autoregressive_prompt(prompt_text)

        generated = self.generate_autoregressive_tokens(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            max_new_tokens=max_new_tokens,
        )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        answers = [text.strip() for text in decoded]
        return answers

    @torch.no_grad()
    def predict_autoregressive_actions(
        self,
        images: dict[str, torch.Tensor],
        task: str,
        state: torch.Tensor,
        max_new_tokens: int = 200,
        control_mode_override: str | None = None,
        end_effector_override: str | None = None,
        image_pad_masks: dict[str, torch.Tensor] | None = None,
        state_memory_mask: torch.Tensor | None = None,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        is_robot_moving: bool = False,
        is_body_moving: bool = False,
    ) -> torch.Tensor:
        """Predict actions using autoregressive generation.

        This method generates actions by autoregressively generating tokens and extracting
        action sequences from the generated tokens.

        Args:
            images: Observation images keyed by camera name.
            task: The executing task description.
            state: The joint state tensor.
            max_new_tokens: Maximum number of new tokens to generate during autoregressive
                inference. Defaults to 200.
            control_mode_override: Optional explicit control mode used during prompt construction.
            end_effector_override: Optional explicit end-effector type used during prompt construction.
            image_pad_masks: Optional temporal image padding masks keyed by
                ``'{image_key}_is_pad'``. This is used when callers pass
                pre-stacked memory images directly.
            state_memory_mask: Optional valid-frame mask for pre-stacked state
                memory. True means the state frame is valid.
            is_robot_moving: Whether action-only mobile-base dimensions are active.
            is_body_moving: Whether the full state-aligned body action prefix is
                active when the mobile base is stationary. This does not change
                the state input dimensions.

        Returns:
            Predicted action tensor.
        """
        ori_device = state.device
        state = state.to(self.device)
        for key in images:
            images[key] = images[key].to(self.device)
        images, memory_pad_masks = self._stack_observation_memory(images)
        memory_pad_masks = self._merge_image_pad_masks(memory_pad_masks, image_pad_masks)

        images, img_masks, image_transform_params = self.image_transform({**images, **memory_pad_masks})
        state = self._quaternion_to_6d_state(state)
        agent_pos_mask = (~torch.isnan(state)).to(dtype=torch.float32)
        if self.state_input_mode == 'proprio_anchor':
            state = torch.nan_to_num(state, nan=0.0)
        state = self.state_normalize_transform(state, embodiment_id=self.embodiment_id)
        if self.state_input_mode == 'proprio_anchor':
            state = state.clamp(-1.0, 1.0)
        action_reference_state = state
        action_state_layout = self._resolve_action_state_layout(
            state,
            is_robot_moving=is_robot_moving,
            is_body_moving=is_body_moving,
        )
        state, agent_pos_mask = self._mask_state_inputs(
            state,
            agent_pos_mask,
            state_input_dim=action_state_layout.state_input_dim,
        )
        current_state = state[-1] if state.ndim == 2 else state
        state_memory, state_memory_masks = self._stack_state_memory(state, state_memory_mask)
        proprioception, anchor_agent_pos_mask = self._prepare_proprio_anchor(state, agent_pos_mask)
        prompt_inputs = {'task': task, **image_transform_params}
        if self.state_input_mode == 'prompt':
            prompt_state = self._mask_state_for_prompt(
                current_state,
                is_robot_moving=is_robot_moving,
                is_body_moving=is_body_moving,
            )
            prompt_inputs['observation.state'] = prompt_state
        if control_mode_override is not None:
            prompt_inputs['control_mode_override'] = control_mode_override
        if end_effector_override is not None:
            prompt_inputs['end_effector_override'] = end_effector_override
        lang_tokens, lang_masks, lang_att_masks, _, _, _, _ = self.prompt_tokenizer_transform(prompt_inputs)

        generated = self.generate_autoregressive_tokens(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            lang_att_masks=lang_att_masks,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            state_memory=state_memory,
            state_memory_masks=state_memory_masks,
            proprioception=proprioception,
            agent_pos_mask=anchor_agent_pos_mask,
        )

        # extract_actions decodes at the FAST-native action dim (auto-inferred inside the
        # FAST tokenizer when original_action_dim is a trimmed/delta-masked dim), then we
        # trim to original_action_dim for the downstream unnorm / absolute transforms.
        pred_action = self.prompt_tokenizer_transform.extract_actions(generated, self.policy.n_action_steps, self.original_action_dim)
        pred_action = pred_action[..., : self.original_action_dim]
        pred_action = pred_action.to(self.device)
        pred_action = self._mask_action_output_dims(pred_action, action_dim=action_state_layout.action_supervised_dim)
        reference_current_state = action_reference_state[-1] if action_reference_state.ndim == 2 else action_reference_state
        state = self.pad_states_and_actions_transform({'observation.state': reference_current_state})['observation.state']
        output_dict = {'action': pred_action[0], 'observation.state': state, 'embodiment_id': self.embodiment_id, 'robot_type': self.robot_type}
        output_dict['observation.state'] = self.state_unnormalize_transform(output_dict['observation.state'], embodiment_id=self.embodiment_id)
        output_dict['action'] = self.action_unnormalize_transform(output_dict['action'], embodiment_id=self.embodiment_id)
        output_dict = self.absolute_actions_transform(output_dict)
        pred_action = output_dict['action'].to(ori_device)
        pred_action = self._mask_action_output_dims(pred_action, action_dim=action_state_layout.action_supervised_dim)
        return pred_action

    @torch.no_grad()
    def generate_autoregressive_tokens(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        lang_att_masks: torch.Tensor | None = None,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        state_memory: torch.Tensor | None = None,
        state_memory_masks: torch.Tensor | None = None,
        proprioception: torch.Tensor | None = None,
        agent_pos_mask: torch.Tensor | None = None,
    ) -> list[list[int]]:
        """Autoregressively generate language tokens until EOS or limit.

        Args:
            images: List of image tensors.
            img_masks: List of image mask tensors.
            lang_tokens: Language token tensors.
            lang_masks: Language mask tensors.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature/top_k/top_p: sampling controls; ``temperature<=0`` => greedy
                argmax (see ``sample_next_token``). Defaults reproduce greedy decoding.

        Returns:
            List of text tokens, one per sample in the batch.
        """
        from .giga_brain_0_utils import sample_next_token

        for i in range(len(images)):
            images[i] = images[i][None, ...]
            img_masks[i] = img_masks[i][None, ...]
        lang_tokens = lang_tokens[None, ...]
        lang_masks = lang_masks[None, ...]
        if lang_att_masks is not None:
            lang_att_masks = lang_att_masks[None, ...]
        if state_memory is not None:
            state_memory = state_memory[None, ...]
        if state_memory_masks is not None:
            state_memory_masks = state_memory_masks[None, ...]

        # Initialize: build prefix cache and get next-step logits
        next_logits, gen_state = self.policy.init_lang_generation(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            lang_att_masks=lang_att_masks,
            emb_ids=torch.tensor([self.embodiment_id], dtype=torch.long, device=self.device),
            state_memory=state_memory,
            state_memory_masks=state_memory_masks,
            proprioception=proprioception,
            agent_pos_mask=agent_pos_mask,
        )

        eos_raw = self.prompt_tokenizer_transform.paligemma_tokenizer.eos_token_id

        eos_ids = list(eos_raw) if isinstance(eos_raw, (list, tuple)) else [eos_raw]
        primary_eos = eos_ids[0]

        generated = []
        bsize = lang_tokens.shape[0]
        finished = torch.zeros(bsize, dtype=torch.bool, device=self.device)
        eos_tensor = torch.tensor(eos_ids, device=self.device)
        # Initialize per-sample generated token lists
        for _ in range(bsize):
            generated.append([])

        for step_idx in range(max_new_tokens):
            step_token = sample_next_token(next_logits, temperature, top_k, top_p).to(torch.long)  # (b,)
            # For finished samples, keep feeding eos to avoid shape mismatch
            step_token = torch.where(finished, torch.tensor(primary_eos, device=step_token.device), step_token)

            # Update each sample
            for i in range(bsize):
                if not finished[i].item():
                    generated[i].append(step_token[i].item())
            # Update finished flags (stop on any eos variant)
            finished = finished | torch.isin(step_token, eos_tensor.to(step_token.device))
            if torch.all(finished):
                break

            # Proceed to the next step
            input_token = step_token.view(bsize, 1)
            next_logits, gen_state = self.policy.next_lang_logits(gen_state, input_token)

        return generated
