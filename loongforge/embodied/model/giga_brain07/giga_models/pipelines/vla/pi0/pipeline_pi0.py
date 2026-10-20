import torch

from ....models import PI0Policy
from ...pipeline import BasePipeline
from .pi0_utils import (
    AbsoluteActions,
    AlohaInputs,
    AlohaOutputs,
    ImageTransform,
    Normalize,
    PadStatesAndActions,
    PromptTokenizerTransform,
    Unnormalize,
)


class Pi0Pipeline(BasePipeline):
    """High-level inference pipeline for PI0 policy.

    This pipeline handles preprocessing (state/image/token), model invocation, and postprocessing back to the environment action space.
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_model_path: str,
        state_norm_stats: dict,
        action_norm_stats: dict,
        original_action_dim: int = 14,
    ) -> None:
        """Initialize the PI0 pipeline.

        Args:
            model_path: Path to the pretrained PI0 policy weights.
            tokenizer_model_path: Path or hub id for the tokenizer used in prompt processing.
            state_norm_stats: Normalization stats for robot state; either {'mean','std'} or {'q01','q99'}.
            action_norm_stats: Normalization stats for actions; either {'mean','std'} or {'q01','q99'}.
            original_action_dim: Dimension of the environment's native action space before padding (e.g., 14 or 16).
        """
        super().__init__()
        self.policy = PI0Policy.from_pretrained(model_path)
        self.policy.eval()
        self.device = 'cpu'
        self.pi05_enabled = self.policy.pi05_enabled
        # Input transforms
        self.aloha_inputs_transform = AlohaInputs()
        self.state_normalize_transform = Normalize(state_norm_stats, use_quantiles=self.pi05_enabled)
        self.pad_states_and_actions_transform = PadStatesAndActions(action_dim=32)
        self.image_transform = ImageTransform(
            resize_imgs_with_padding=(224, 224),
            enable_image_aug=False,
        )
        max_length = 200 if self.pi05_enabled else 48
        self.prompt_tokenizer_transform = PromptTokenizerTransform(
            tokenizer_model_path=tokenizer_model_path, max_length=max_length, discrete_state_input=self.pi05_enabled
        )
        # Output transforms
        self.state_unnormalize_transform = Unnormalize(state_norm_stats, use_quantiles=self.pi05_enabled)
        self.action_unnormalize_transform = Unnormalize(action_norm_stats, use_quantiles=self.pi05_enabled)
        self.absolute_actions_transform = AbsoluteActions()
        self.aloha_outputs_transform = AlohaOutputs(original_action_dim=original_action_dim)

    def to(self, device: torch.device | str):
        """Move the policy and all transforms to the specified device.

        Args:
            device: Target device (e.g., 'cuda', 'cpu', torch.device('cuda:0')).

        Returns:
            self: Enables chained calls.
        """
        self.device = device
        self.policy.to(device)
        self.aloha_inputs_transform.to(device)
        self.state_normalize_transform.to(device)
        self.state_unnormalize_transform.to(device)
        self.action_unnormalize_transform.to(device)
        self.absolute_actions_transform.to(device)
        self.aloha_outputs_transform.to(device)
        return self

    def compile(self, **kwargs) -> None:
        """Compile the sampling function for improved runtime speed.

        Note:
            This uses `torch.compile` under the hood and forwards any kwargs.
        """
        self.policy.sample_actions = torch.compile(self.policy.sample_actions, **kwargs)

    @torch.no_grad()
    def __call__(
        self,
        images: dict[str, torch.Tensor],
        task: str,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Run one forward pass from raw inputs to final action sequence.

        Args:
            images: Observation images of the robot. Each value is a tensor with shape (C,H,W).
            task: Natural language task description.
            state: The robot joint state tensor with shape (state_dim,).

        Returns:
            A tensor of predicted actions with shape (num_steps, original_action_dim) on the original input device.
        """
        # Input transforms
        ori_device = state.device
        state = state.to(self.device)
        for key in images:
            images[key] = images[key].to(self.device)

        state = self.aloha_inputs_transform({'observation.state': state})['observation.state']
        state = self.state_normalize_transform(state)
        images, img_masks = self.image_transform(images)
        lang_tokens, lang_masks = self.prompt_tokenizer_transform({'task': task, 'observation.state': state})
        state = self.pad_states_and_actions_transform({'observation.state': state})['observation.state']

        state = state[None, ...]
        for i in range(len(images)):
            images[i] = images[i][None, ...]
            img_masks[i] = img_masks[i][None, ...]
        lang_tokens = lang_tokens[None, ...]
        lang_masks = lang_masks[None, ...]

        # Inference
        pred_action = self.policy.sample_actions(images, img_masks, lang_tokens, lang_masks, state=state)

        # Output transforms
        output_dict = {'action': pred_action[0], 'observation.state': state[0]}
        output_dict['observation.state'] = self.state_unnormalize_transform(output_dict['observation.state'])
        output_dict['action'] = self.action_unnormalize_transform(output_dict['action'])
        output_dict = self.absolute_actions_transform(output_dict)
        pred_action = self.aloha_outputs_transform(output_dict)['action'].to(ori_device)

        return pred_action
