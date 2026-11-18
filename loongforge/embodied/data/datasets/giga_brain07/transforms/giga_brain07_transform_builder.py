# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""GigaBrain07 per-sample transform builder (shared-pipeline registration).

Registers ``@register_transform_builder("giga_brain07")`` so the shared
``build_transforms_from_args`` pipeline constructs the giga per-sample transform,
mirroring xvla's ``build_xvla_transforms``. This module is intentionally light:
it imports only the registry at module load and defers the heavy
``GigaBrain07Transform`` import to builder call time, so importing the
giga_brain07 transforms package (e.g. when ``train/config_map.py`` pulls the slim
``GigaBrain07DataConfig``) does not drag in the transform's model-side building
blocks.
"""

from loongforge.embodied.data.datasets.transforms.registry import (
    register_transform_builder,
)


def _build_giga_brain07_transform_cfg(data_cfg) -> dict:
    """Assemble the GigaBrain07Transform kwargs from a typed DataConfig.

    Extracted verbatim from the former ``GigaBrain07Trainer._build_dataloaders``
    (field mapping unchanged) so the transform is built through the shared
    pipeline rather than hand-constructed in the trainer.
    """
    d = data_cfg
    return dict(
        is_train=bool(d.is_train),
        state_input_mode=d.state_input_mode,
        observation_memory_size=int(d.observation_memory_size),
        agent_pos_config={"state": int(d.agent_pos_state)},
        use_quaternion_to_6d=bool(d.use_quaternion_to_6d),
        robot_type_embodiment_id_overrides={d.robot_type: d.embodiment_override},
        delta_action_cfg=dict(
            selector="embodiment_id",
            use_delta_joint_actions=bool(d.use_delta_joint_actions),
            mask_unsupervised_action_dims_for_noise=bool(
                d.mask_unsupervised_action_dims_for_noise
            ),
            mask={d.embodiment_id: list(d.delta_mask)},
        ),
        norm_cfg=dict(
            norm_stats_path={d.embodiment_id: d.norm_stats_path},
            use_quantiles=bool(d.use_quantiles),
        ),
        image_cfg=dict(
            resize_imgs_with_padding=list(d.resize_hw),
            enable_image_aug=bool(d.enable_image_aug),
            present_img_keys=list(d.camera_keys),
            enable_depth_img=bool(d.enable_depth_img),
            vlm_type=d.vlm_type,
        ),
        prompt_cfg=dict(
            tokenizer_model_path=d.tokenizer_model_path,
            fast_tokenizer_path=d.fast_tokenizer_path,
            enable_control_mode_token=bool(d.enable_control_mode_token),
            enable_end_effector_token=bool(d.enable_end_effector_token),
            max_length=int(d.max_length),
            state_input_mode=d.state_input_mode,
            discrete_state_input=bool(d.discrete_state_input),
            discrete_state_input_for_pose_embodiments=bool(
                d.discrete_state_input_for_pose_embodiments
            ),
            encode_action_input=bool(d.encode_action_input),
            encode_sub_task_input=bool(d.encode_sub_task_input),
            encoded_action_horizon=int(d.encoded_action_horizon),
            sample_ratios=dict(d.sample_ratios),
            vlm_type=d.vlm_type,
        ),
    )


@register_transform_builder("giga_brain07")
def build_giga_brain07_transforms(ctx):
    """Per-sample transform builder for giga_brain07 (xvla-style).

    Returns a single ``GigaBrain07Transform`` built from the typed DataConfig,
    so the shared ``build_transforms_from_args`` pipeline attaches it exactly as
    it does for xvla / pi05. The ``model_type`` guard mirrors
    ``build_xvla_transforms``; ``GigaBrain07Transform`` is imported lazily so this
    module stays light at registration time.
    """
    if getattr(ctx.model_cfg, "model_type", None) != "giga_brain07":
        return []
    from loongforge.embodied.data.datasets.giga_brain07.transforms.giga_brain07_transform import (
        GigaBrain07Transform,
    )

    return [GigaBrain07Transform(**_build_giga_brain07_transform_cfg(ctx.data_cfg))]
