import os
from typing import List, Optional, Union

import numpy as np
import torch
import torchvision.transforms.functional as F
from diffusers import DiffusionPipeline
from diffusers.image_processor import PipelineImageInput
from diffusers.models import AutoencoderKLWan
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from einops import rearrange

from ....exports.transformer_engine import apply_fp8_autowrap
from ....models.diffusion.giga_world_0 import GigaWorld0Transformer3DModel, T5TextEncoder
from ....schedulers import EDMRESMultistepScheduler
from ....utils import download_from_huggingface


class GigaWorld0Pipeline(DiffusionPipeline):
    """
    Diffusion pipeline for GigaWorld0: handles text-to-video generation, encoding/decoding, and denoising loop.
    This class integrates text encoding, transformer, VAE, and scheduler to generate videos from text prompts.
    """

    def __init__(
        self,
        text_encoder: T5TextEncoder,
        transformer: GigaWorld0Transformer3DModel,
        vae: AutoencoderKLWan,
        scheduler: EDMRESMultistepScheduler,
    ):
        """Initialize the GigaWorld0Pipeline with all required modules.

        Args:
            text_encoder: Module to encode text prompts.
            transformer: Main transformer model for video generation.
            vae: Variational autoencoder for encoding/decoding video frames.
            scheduler: Scheduler for diffusion process.
        """
        self.register_modules(
            text_encoder=text_encoder,
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
        )
        # Compute scale factors for VAE
        self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample) if getattr(self, 'vae', None) else 4
        self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample) if getattr(self, 'vae', None) else 8
        self.latent_channels = self.vae.config.z_dim
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    @classmethod
    def from_pretrained(
        cls,
        transformer_model_path,
        text_encoder_model_path=None,
        vae_model_path=None,
        lora_model_path=None,
        lora_fuse=False,
        fp8_eval=True,
    ):
        """Load a pretrained pipeline from disk or HuggingFace Hub.

        Args:
            transformer_model_path: Path to the transformer model.
            text_encoder_model_path: Path to the text encoder (optional).
            vae_model_path: Path to the VAE (optional).
            lora_model_path: Path to LoRA weights (optional).
            lora_fuse: Whether to fuse LoRA weights.
            fp8_eval: Whether to use FP8 evaluation.
        Returns:
            An instance of GigaWorld0Pipeline.
        """
        if text_encoder_model_path is None:
            text_encoder_model_path = download_from_huggingface(
                'google-t5/t5-11b',
                local_dir=os.path.join(transformer_model_path, '..', 'text_encoder'),
            )
        if vae_model_path is None:
            vae_model_path = download_from_huggingface(
                'Wan-AI/Wan2.1-T2V-1.3B-Diffusers',
                local_dir=os.path.join(transformer_model_path, '..'),
                folders='vae',
            )
        transformer = GigaWorld0Transformer3DModel.from_pretrained(transformer_model_path)
        transformer.to(torch.bfloat16)
        text_encoder = T5TextEncoder(text_encoder_model_path)
        vae = AutoencoderKLWan.from_pretrained(vae_model_path)
        vae.to(torch.bfloat16)
        scheduler = EDMRESMultistepScheduler(
            prediction_type='rf',
            solver_order=1,
            final_sigmas_type='sigma_min',
            sigma_data=1.0,
        )
        if lora_model_path is not None:
            transformer.load_lora(lora_model_path, fuse=lora_fuse)
        if transformer.fp8:
            apply_fp8_autowrap(transformer, use_autocast_during_eval=fp8_eval)
        pipe = cls(
            text_encoder=text_encoder,
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
        )
        return pipe

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
    ):
        """Encode text and negative prompts into embeddings for the
        transformer.

        Args:
            prompt: Text prompt(s) to encode.
            negative_prompt: Negative prompt(s) for classifier-free guidance.
            prompt_embeds: Precomputed prompt embeddings (optional).
            negative_prompt_embeds: Precomputed negative prompt embeddings (optional).
        Returns:
            Tuple of (prompt_embeds, negative_prompt_embeds)
        """
        if prompt_embeds is None:
            if isinstance(prompt, str):
                prompt = [prompt]
            prompt_embeds = self.text_encoder.encode_prompts(prompt)
        if self.do_classifier_free_guidance:
            if negative_prompt_embeds is None:
                if negative_prompt is None:
                    negative_prompt = [''] * prompt_embeds.shape[0]
                elif isinstance(negative_prompt, str):
                    negative_prompt = [negative_prompt]
                negative_prompt_embeds = self.text_encoder.encode_prompts(negative_prompt)
        return prompt_embeds, negative_prompt_embeds

    def encode(self, video):
        """Encode a video tensor into latent space using the VAE.

        Args:
            video: Input video tensor.
        Returns:
            Latent representation of the video.
        """
        video = video.to(self.vae.dtype)
        latents = self.vae.encode(video).latent_dist.mode()
        latents_mean, latents_std = self.vae.config.latents_mean, self.vae.config.latents_std
        latents_mean = torch.tensor(latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(latents)
        latents_std = torch.tensor(latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(latents)
        latents = (latents - latents_mean) / latents_std * self.scheduler.config.sigma_data
        return latents

    def decode(self, latents):
        """Decode latent representation back to video frames using the VAE.

        Args:
            latents: Latent tensor to decode.
        Returns:
            Decoded video tensor.
        """
        latents = latents.to(self.vae.dtype)
        latents_mean, latents_std = self.vae.config.latents_mean, self.vae.config.latents_std
        latents_mean = torch.tensor(latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(latents)
        latents_std = torch.tensor(latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(latents)
        latents = latents * latents_std / self.scheduler.config.sigma_data + latents_mean
        video = self.vae.decode(latents, return_dict=False)[0]
        return video

    def prepare_cond_latents(
        self,
        image: PipelineImageInput,
        num_frames: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """Prepare conditional latents and masks for input images (for video
        inpainting or conditioning).

        Args:
            image: Input image(s) or tensor.
            num_frames: Number of frames in the video.
            height: Target height.
            width: Target width.
            device: Device to use.
            dtype: Data type.
        Returns:
            Tuple of (cond_latents, cond_masks)
        """
        if image is None:
            images = []
        elif isinstance(image, list) or isinstance(image, torch.Tensor):
            images = image
        else:
            images = [image]
        if len(images) == 0:
            latents_shape = [
                1,
                self.latent_channels,
                (num_frames - 1) // self.vae_scale_factor_temporal + 1,
                height // self.vae_scale_factor_spatial,
                width // self.vae_scale_factor_spatial,
            ]
            cond_latents = torch.zeros(latents_shape, device=device, dtype=dtype)
            cond_masks = torch.zeros((1, 1, cond_latents.shape[2], 1, 1), device=device, dtype=dtype)
        else:
            num_images = len(images)
            assert num_images <= num_frames and (num_images - 1) % self.vae_scale_factor_temporal == 0
            num_cond_frames = 1 + (num_images - 1) // self.vae_scale_factor_temporal
            if isinstance(images, torch.Tensor):
                cond_images = 2.0 * images - 1.0
            else:
                cond_images = [np.array(image) for image in images]
                cond_images = np.stack(cond_images, axis=0)
                cond_images = rearrange(cond_images, 't h w c -> t c h w')
                cond_images = cond_images / 127.5 - 1.0
                cond_images = torch.from_numpy(cond_images)
            cond_images = cond_images.to(device, dtype)
            cond_images = F.resize(
                cond_images,
                size=(height, width),  # type: ignore
                interpolation=F.InterpolationMode.BICUBIC,
                antialias=True,
            )
            last_image = torch.zeros_like(cond_images[-1:])
            last_images = last_image.repeat(num_frames - cond_images.shape[0], 1, 1, 1)
            cond_images = torch.cat([cond_images, last_images], dim=0)
            cond_images = rearrange(cond_images, 't c h w -> 1 c t h w')
            cond_latents = self.encode(cond_images).to(dtype)
            cond_masks = torch.zeros((1, 1, cond_latents.shape[2], 1, 1), device=device, dtype=dtype)
            cond_masks[:, :, :num_cond_frames] = 1
        return cond_latents, cond_masks

    @property
    def guidance_scale(self):
        """Get the current guidance scale for classifier-free guidance."""
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        """Whether classifier-free guidance is enabled (guidance_scale >
        1.0)."""
        return self._guidance_scale > 1.0

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        image: PipelineImageInput = None,
        guidance_scale: float = 7.0,
        num_inference_steps: int = 30,
        fps: int = 16,
        num_frames: int = 93,
        height: int = 480,
        width: int = 768,
        seed: int = -1,
        augment_sigma: float = 0.001,
        sigma_max: float = 80.0,
        output_type: Optional[str] = 'pil',
    ):
        """Run the diffusion pipeline to generate a video from a text prompt.

        Args:
            prompt: Text prompt for generation.
            negative_prompt: Negative prompt for classifier-free guidance.
            image: Optional conditioning image(s).
            guidance_scale: Classifier-free guidance scale.
            num_inference_steps: Number of diffusion steps.
            fps: Frames per second for output video.
            num_frames: Number of frames to generate.
            height: Output video height.
            width: Output video width.
            seed: Random seed for reproducibility.
            augment_sigma: Sigma for conditioning augmentation.
            sigma_max: Maximum sigma for scheduler.
            output_type: Output format ('pil', 'latent', etc.).
        Returns:
            Generated video in the specified format.
        """
        self._guidance_scale = guidance_scale
        batch_size = 1
        device = self._execution_device
        dtype = self.transformer.dtype

        generator = None
        if seed > 0:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(prompt, negative_prompt=negative_prompt)
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        prompt_embeds = prompt_embeds.to(dtype)

        self.scheduler.config.sigma_max = sigma_max
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        sigmas = self.scheduler.sigmas.to(device)

        # Prepare latent variables
        shape = (
            batch_size,
            self.latent_channels,
            (num_frames - 1) // self.vae_scale_factor_temporal + 1,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = latents * self.scheduler.init_noise_sigma

        if image is not None:
            cond_latents, cond_masks = self.prepare_cond_latents(
                image,
                num_frames,
                height,
                width,
                device=device,
                dtype=torch.float32,
            )
            cond_sigma = torch.tensor([augment_sigma], device=device, dtype=cond_latents.dtype)
            masks_input = cond_masks.repeat(batch_size, 1, 1, cond_latents.shape[-2], cond_latents.shape[-1])
        else:
            cond_latents = cond_masks = None

        padding_mask = torch.zeros(1, 1, height, width, device=device, dtype=dtype)
        if self.do_classifier_free_guidance:
            padding_mask = torch.cat([padding_mask, padding_mask], dim=0)

        # Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = latents
                if cond_latents is not None:
                    cond_noise = torch.randn(
                        cond_latents.shape,
                        device=device,
                        dtype=torch.float32,
                        generator=generator,
                    )
                    latent_model_input = self.scheduler.add_condition_inputs(
                        latent_model_input,
                        sigmas[i],
                        cond_sample=cond_latents,
                        cond_mask=cond_masks,
                        cond_noise=cond_noise,
                        cond_sigma=cond_sigma,
                    )
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                timestep = t
                if cond_masks is not None:
                    latent_model_input = torch.cat([latent_model_input, masks_input], dim=1)
                    timestep = timestep.view(1, 1, 1, 1, 1).expand(latents.size(0), -1, latents.size(2), -1, -1)
                    t_conditioning = cond_sigma / (cond_sigma + 1)
                    cond_timestep = cond_masks * t_conditioning + (1 - cond_masks) * timestep
                    timestep = cond_timestep.to(dtype)
                if self.do_classifier_free_guidance:
                    latent_model_input = torch.cat([latent_model_input] * 2)
                latent_model_input = latent_model_input.to(dtype)

                timestep = timestep.expand(latent_model_input.shape[0], -1, -1, -1, -1)
                timestep = timestep.to(dtype)

                noise_pred = self.transformer(
                    x=latent_model_input,
                    timesteps=timestep,
                    crossattn_emb=prompt_embeds,
                    fps=fps,
                    padding_mask=padding_mask,
                )

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_text + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                noise_pred = noise_pred.float()

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred,
                    timestep=t,
                    sample=latents,
                    cond_sample=cond_latents,
                    cond_mask=cond_masks,
                    return_dict=False,
                )[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or (i + 1) % self.scheduler.order == 0:
                    progress_bar.update()

        if not output_type == 'latent':
            video = self.decode(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        return video
