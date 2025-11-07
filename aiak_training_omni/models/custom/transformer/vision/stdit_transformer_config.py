"""custom transformer config for stdit model"""

from dataclasses import dataclass

from megatron.core.transformer.transformer_config import TransformerConfig


@dataclass
class StditTransformerConfig(TransformerConfig):
    """configuration class for the stdit transformer"""

    num_latent_frames: int = 0
    """Number of frames in the latent."""

    max_latent_height: int = 0
    """Maximum height of the latent."""

    max_latent_width: int = 0
    """Maximum width of the latent."""

    latent_in_channels: int = 0
    """Number of channels in the input latent."""

    latent_out_channels: int = 0
    """Number of channels in the output latent."""

    latent_patch_size: tuple = (1, 1, 1)
    """Patch size of the latent."""

    latent_space_scale: float = 1.0
    """Space scale of the latent."""

    latent_time_scale: float = 1.0
    """Time scale of the latent."""

    caption_channels: int = 0
    """Number of channels in the caption."""

    max_text_length: int = 0
    """Maximum text length."""

    max_image_length: int = 0
    """Maximum image length."""

    max_video_length: int = 0
    """Maximum video length."""

    def __post_init__(self):
        super().__post_init__()

        if len(self.latent_patch_size) != 3:
            raise ValueError(
                f"latent_patch_size: {self.latent_patch_size} must have three dimensions."
            )
        if self.latent_patch_size[1] != self.latent_patch_size[2]:
            raise ValueError(
                f"latent_patch_size: {self.latent_patch_size} must have equal height and width."
            )
        if self.latent_time_scale is None:
            self.latent_time_scale = 1.0 / self.latent_patch_size[0]

        elif self.latent_time_scale != 1.0 / self.latent_patch_size[0]:
            raise ValueError(
                f"latent_time_scale: {self.latent_time_scale} must be 1.0 / latent_patch_size[0]."
            )

        if self.latent_space_scale is None:
            self.latent_space_scale = 1.0 / self.latent_patch_size[1]
        elif self.latent_space_scale != 1.0 / self.latent_patch_size[1]:
            raise ValueError(
                f"latent_space_scale: {self.latent_space_scale} must be 1.0 / latent_patch_size[1]."
            )
