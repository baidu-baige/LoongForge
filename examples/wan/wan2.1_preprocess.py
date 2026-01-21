"""wan2.1 preprocess script"""

import torch, os, imageio, argparse
from torchvision.transforms import v2
from einops import rearrange
import lightning as pl
import pandas as pd
from diffsynth import WanVideoPipeline, ModelManager, load_state_dict
from peft import LoraConfig, inject_adapter_in_model
import torchvision
from PIL import Image
import numpy as np
import socket
import json
import inspect


def dump(x, name="tensor.txt", line_number=inspect.currentframe().f_lineno, pp=2, cp=1):
    """Dump tensor to file"""
    tensor_numpy = x.detach().cpu().float().reshape(-1).numpy()
    np.savetxt(
        f"{name}_line_{line_number}.txt_{pp}_{cp}_{torch.cuda.current_device()}",
        tensor_numpy,
        fmt="%.4f",
    )


class TextVideoDataset(torch.utils.data.Dataset):
    """TextVideoDataset"""
    def __init__(
        self,
        base_path,
        metadata_path,
        max_num_frames=81,
        frame_interval=1,
        num_frames=81,
        height=480,
        width=832,
        is_i2v=False,
    ):
        metadata = []
        with open(metadata_path, "r") as f:
            for line in f:
                metadata.append(json.loads(line))
        self.path = [data["video_path"].replace("pfs/", "") for data in metadata]
        self.text = [data["dense_lang"] for data in metadata]

        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.is_i2v = is_i2v

        self.frame_process = v2.Compose(
            [
                v2.CenterCrop(size=(height, width)),
                v2.Resize(size=(height, width), antialias=True),
                v2.ToTensor(),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def crop_and_resize(self, image):
        """
        Scale image proportionally and crop to target size
        """
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height * scale), round(width * scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        )
        return image

    def load_frames_using_imageio(
        self,
        file_path,
        max_num_frames,
        start_frame_id,
        interval,
        num_frames,
        frame_process,
    ):
        """Load video frames"""
        reader = imageio.get_reader(file_path)
        if (
            reader.count_frames() < max_num_frames
            or reader.count_frames() - 1 < start_frame_id + (num_frames - 1) * interval
        ):
            # Get the last frame of the video
            last_frame_id = reader.count_frames() - 1
            last_frame = reader.get_data(last_frame_id)
            last_frame = Image.fromarray(last_frame)
            last_frame = self.crop_and_resize(last_frame)
            last_frame_processed = self.frame_process(last_frame)

            # Create a list of frames with the existing frames
            frames = []
            first_frame = None
            for frame_id in range(min(num_frames, reader.count_frames())):
                frame = reader.get_data(
                    min(start_frame_id + frame_id * interval, last_frame_id)
                )
                frame = Image.fromarray(frame)
                frame = self.crop_and_resize(frame)
                if first_frame is None:
                    first_frame = frame
                frame = frame_process(frame)
                frames.append(frame)

            # Fill the remaining frames with the last frame
            while len(frames) < num_frames:
                frames.append(last_frame_processed)

            reader.close()

            frames = torch.stack(frames, dim=0)
            frames = rearrange(frames, "T C H W -> C T H W")

            first_frame = v2.functional.center_crop(
                first_frame, output_size=(self.height, self.width)
            )
            first_frame = np.array(first_frame)

            if self.is_i2v:
                return frames, first_frame
            else:
                return frames

        frames = []
        first_frame = None
        for frame_id in range(num_frames):
            frame = reader.get_data(start_frame_id + frame_id * interval)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            if first_frame is None:
                first_frame = frame
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")

        first_frame = v2.functional.center_crop(
            first_frame, output_size=(self.height, self.width)
        )
        first_frame = np.array(first_frame)

        if self.is_i2v:
            return frames, first_frame
        else:
            return frames

    def load_video(self, file_path):
        """Load_video"""
        start_frame_id = torch.randint(
            0, self.max_num_frames - (self.num_frames - 1) * self.frame_interval, (1,)
        )[0]
        frames = self.load_frames_using_imageio(
            file_path,
            self.max_num_frames,
            start_frame_id,
            self.frame_interval,
            self.num_frames,
            self.frame_process,
        )
        return frames

    def is_image(self, file_path):
        """Return if is image"""
        file_ext_name = file_path.split(".")[-1]
        if file_ext_name.lower() in ["jpg", "jpeg", "png", "webp"]:
            return True
        return False

    def load_image(self, file_path):
        """Load image"""
        frame = Image.open(file_path).convert("RGB")
        frame = self.crop_and_resize(frame)
        first_frame = frame
        frame = self.frame_process(frame)
        frame = rearrange(frame, "C H W -> C 1 H W")
        return frame

    def __getitem__(self, data_id):
        """Get dataset item"""
        text = self.text[data_id]
        path = self.path[data_id]
        if self.is_image(path):
            if self.is_i2v:
                raise ValueError(
                    f"{path} is not a video. I2V model doesn't support image-to-image training."
                )
            video = self.load_image(path)
        else:
            video = self.load_video(path)
        if self.is_i2v:
            video, first_frame = video
            data = {
                "text": text,
                "video": video,
                "path": path,
                "first_frame": first_frame,
            }
        else:
            data = {"text": text, "video": video, "path": path}
        return data

    def __len__(self):
        """Dataset length"""
        return len(self.path)


class LightningModelForDataProcess(pl.LightningModule):
    """LightningModelForDataProcess"""
    def __init__(
        self,
        text_encoder_path,
        vae_path,
        image_encoder_path=None,
        tiled=False,
        tile_size=(34, 34),
        tile_stride=(18, 16),
    ):
        """LightningModelForDataProcess init"""
        super().__init__()
        model_path = [text_encoder_path, vae_path]
        if image_encoder_path is not None:
            model_path.append(image_encoder_path)
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models(model_path)
        self.pipe = WanVideoPipeline.from_model_manager(model_manager)

        self.pipe.scheduler.set_timesteps(1000, training=True)
        torch.manual_seed(10086)
        self.tiler_kwargs = {
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
        }

    def test_step(self, batch, batch_idx):
        """Write processed data to file"""
        print(f"batch_idx: {batch_idx}")
        text, video, path = batch["text"][0], batch["video"], batch["path"][0]

        self.pipe.device = self.device
        if video is not None:
            # prompt
            prompt_emb = self.pipe.encode_prompt(text)
            # video
            video = video.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            latents = self.pipe.encode_video(video, **self.tiler_kwargs)[0]
            # image
            if "first_frame" in batch:
                first_frame = Image.fromarray(batch["first_frame"][0].cpu().numpy())
                _, _, num_frames, height, width = video.shape
                image_emb = self.pipe.encode_image(
                    first_frame, None, num_frames, height, width
                )
            else:
                image_emb = {}
            data = {
                "latents": latents,
                "prompt_emb": prompt_emb,
                "image_emb": image_emb,
            }

            self.pipe.device = self.device
            noise = torch.randn_like(latents)
            timestep_id = torch.randint(
                0, self.pipe.scheduler.num_train_timesteps, (1,)
            )
            timestep = self.pipe.scheduler.timesteps[timestep_id].to(
                dtype=self.pipe.torch_dtype, device=self.pipe.device
            )  # 非等距的
            extra_input = self.pipe.prepare_extra_input(latents)
            noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
            training_target = self.pipe.scheduler.training_target(
                latents, noise, timestep
            )
            scale = self.pipe.scheduler.training_weight(timestep)

            data = {
                "latents": latents,
                "prompt_emb": prompt_emb,
                "image_emb": image_emb,
            }
            print(f"scale: {scale.dtype}, scale: {scale}, timestep: {timestep}")
            torch.save(data, path + ".tensors.pth")


class TensorDataset(torch.utils.data.Dataset):
    """TensorDataset"""
    def __init__(self, base_path, metadata_path, steps_per_epoch):
        """Dataset init"""
        metadata = []
        with open(metadata_path, "r") as f:
            for line in f:
                metadata.append(json.loads(line))
        # self.path = [os.path.join(base_path, data["video_path"].split("/")[-1]) for data in metadata]
        self.path = [data["video_path"] for data in metadata]

        print(len(self.path), "videos in metadata.")
        self.path = [
            i + ".tensors.pth" for i in self.path if os.path.exists(i + ".tensors.pth")
        ]
        print(len(self.path), "tensors cached in metadata.")
        assert len(self.path) > 0

        self.steps_per_epoch = steps_per_epoch
        # torch.manual_seed(10086)  # jhc_debug

    def __getitem__(self, index):
        """Dataset get item"""
        data_id = torch.randint(0, len(self.path), (1,))[0]
        data_id = (data_id + index) % len(self.path)  # For fixed seed.
        # data_id = index
        data_id = index
        data_id = 0
        print(f"get_dataset: {data_id}")
        path = self.path[data_id]
        print(f"path {path}")
        data = torch.load(path, weights_only=True, map_location="cpu")
        return data

    def __len__(self):
        """Dataset length"""
        return self.steps_per_epoch


class LightningModelForTrain(pl.LightningModule):
    """LightningModelForTrain"""
    def __init__(
        self,
        dit_path,
        learning_rate=1e-5,
        lora_rank=4,
        lora_alpha=4,
        train_architecture="lora",
        lora_target_modules="q,k,v,o,ffn.0,ffn.2",
        init_lora_weights="kaiming",
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        pretrained_lora_path=None,
    ):
        """Lightning model init"""
        super().__init__()
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        if os.path.isfile(dit_path):
            model_manager.load_models([dit_path])
        else:
            dit_path = dit_path.split(",")
            model_manager.load_models([dit_path])

        self.pipe = WanVideoPipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.freeze_parameters()
        if train_architecture == "lora":
            self.add_lora_to_model(
                self.pipe.denoising_model(),
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_target_modules=lora_target_modules,
                init_lora_weights=init_lora_weights,
                pretrained_lora_path=pretrained_lora_path,
            )
        else:
            self.pipe.denoising_model().requires_grad_(True)

        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload

    def freeze_parameters(self):
        """Freeze parameters"""
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()

    def add_lora_to_model(
        self,
        model,
        lora_rank=4,
        lora_alpha=4,
        lora_target_modules="q,k,v,o,ffn.0,ffn.2",
        init_lora_weights="kaiming",
        pretrained_lora_path=None,
        state_dict_converter=None,
    ):
        """Add lora to model"""
        self.lora_alpha = lora_alpha
        if init_lora_weights == "kaiming":
            init_lora_weights = True

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules.split(","),
        )
        model = inject_adapter_in_model(lora_config, model)
        for param in model.parameters():
            # Upcast LoRA parameters into fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)

        # Lora pretrained lora weights
        if pretrained_lora_path is not None:
            state_dict = load_state_dict(pretrained_lora_path)
            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)
            missing_keys, unexpected_keys = model.load_state_dict(
                state_dict, strict=False
            )
            all_keys = [i for i, _ in model.named_parameters()]
            num_updated_keys = len(all_keys) - len(missing_keys)
            num_unexpected_keys = len(unexpected_keys)
            print(
                f"{num_updated_keys} parameters are loaded from "
                f"{pretrained_lora_path}.{num_unexpected_keys} parameters are unexpected."
            )

    def training_step(self, batch, batch_idx):
        """Training step"""
        # Data
        latents = batch["latents"].to(self.device)
        prompt_emb = batch["prompt_emb"]
        prompt_emb["context"] = prompt_emb["context"][0].to(self.device)
        image_emb = batch["image_emb"]
        if "clip_feature" in image_emb:
            image_emb["clip_feature"] = image_emb["clip_feature"][0].to(self.device)
        if "y" in image_emb:
            image_emb["y"] = image_emb["y"][0].to(self.device)

        noisy_latents = latents
        timestep = batch["timestep"][0]
        training_target = batch["training_target"]
        scale = batch["scale"]
        extra_input = {}
        print(f"timestep: {timestep}, scale: {scale}")

        noise_pred = self.pipe.denoising_model()(
            noisy_latents,
            timestep=timestep,
            **prompt_emb,
            **extra_input,
            **image_emb,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload,
        )
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        # loss = loss * self.pipe.scheduler.training_weight(timestep)
        loss = loss * scale
        print(f"jhc_loss: {loss}")
        # Record log
        self.log("train_loss_returned", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Optimizer config"""
        trainable_modules = filter(
            lambda p: p.requires_grad, self.pipe.denoising_model().parameters()
        )
        optimizer = torch.optim.AdamW(trainable_modules, lr=1e-4, weight_decay=1e-2)
        print("learning rate: ", self.learning_rate)
        return optimizer

    def on_save_checkpoint(self, checkpoint):
        """Save checkpoint"""
        checkpoint.clear()
        trainable_param_names = list(
            filter(
                lambda named_param: named_param[1].requires_grad,
                self.pipe.denoising_model().named_parameters(),
            )
        )
        trainable_param_names = set(
            [named_param[0] for named_param in trainable_param_names]
        )
        state_dict = self.pipe.denoising_model().state_dict()
        lora_state_dict = {}
        for name, param in state_dict.items():
            if name in trainable_param_names:
                lora_state_dict[name] = param
        checkpoint.update(lora_state_dict)


def parse_args():
    """Parse arguments"""
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--task",
        type=str,
        default="data_process",
        required=True,
        choices=["data_process", "train"],
        help="Task. `data_process` or `train`.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        required=True,
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default=None,
        help="Path of text encoder.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        help="Path of image encoder.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help="Path of VAE.",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default=None,
        help="Path of DiT.",
    )
    parser.add_argument(
        "--tiled",
        default=False,
        action="store_true",
        help="Whether enable tile encode in VAE. This option can reduce VRAM required.",
    )
    parser.add_argument(
        "--tile_size_height",
        type=int,
        default=34,
        help="Tile size (height) in VAE.",
    )
    parser.add_argument(
        "--tile_size_width",
        type=int,
        default=34,
        help="Tile size (width) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_height",
        type=int,
        default=18,
        help="Tile stride (height) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_width",
        type=int,
        default=16,
        help="Tile stride (width) in VAE.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=500,
        help="Number of steps per epoch.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Image width.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=15,
        help="Number of subprocesses to use for data loading."
        "0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="The number of batches in gradient accumulation.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q,k,v,o,ffn.0,ffn.2",
        help="Layers with LoRA modules.",
    )
    parser.add_argument(
        "--init_lora_weights",
        type=str,
        default="kaiming",
        choices=["gaussian", "kaiming"],
        help="The initializing method of LoRA weight.",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="auto",
        choices=[
            "auto",
            "deepspeed_stage_1",
            "deepspeed_stage_2",
            "deepspeed_stage_3",
            "deepspeed_stage_2_offload",
            "deepspeed_stage_3_offload",
            "fsdp",
            "fsdp_native",
        ],
        help="Training strategy",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=4.0,
        help="The weight of the LoRA update matrices.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing_offload",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing offload.",
    )
    parser.add_argument(
        "--train_architecture",
        type=str,
        default="lora",
        choices=["lora", "full"],
        help="Model structure to train. LoRA training or full training.",
    )
    parser.add_argument(
        "--pretrained_lora_path",
        type=str,
        default=None,
        help="Pretrained LoRA path. Required if the training is resumed.",
    )
    parser.add_argument(
        "--use_swanlab",
        default=False,
        action="store_true",
        help="Whether to use SwanLab logger.",
    )
    parser.add_argument(
        "--swanlab_mode",
        default=None,
        help="SwanLab mode (cloud or local).",
    )
    # Multi-node training parameters
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="Number of nodes (machines) to use for distributed training.",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Number of GPUs per node to use for training. If None, will use all available GPUs.",
    )
    parser.add_argument(
        "--node_rank",
        type=int,
        default=0,
        help="Rank of the current node in distributed training (0 to num_nodes-1).",
    )
    parser.add_argument(
        "--master_addr",
        type=str,
        default="localhost",
        help="Address of the master node for distributed training.",
    )
    parser.add_argument(
        "--master_port",
        type=str,
        default="12355",
        help="Port on the master node for communication during distributed training.",
    )
    parser.add_argument(
        "--syncbnorm",
        default=False,
        action="store_true",
        help="Whether to use synchronized batch normalization across devices.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from.",
    )
    parser.add_argument(
        "--fsdp_config",
        type=str,
        default=None,
        help="Path to a FSDP config file in JSON format.",
    )
    parser.add_argument(
        "--checkpoint_every_n_steps",
        type=int,
        default=None,
        help="Save a checkpoint every N training steps (in addition to end of epoch).",
    )

    args = parser.parse_args()
    return args


def data_process(args):
    """Data processing pipeline."""
    dataset = TextVideoDataset(
        args.dataset_path,
        os.path.join(args.dataset_path, "metadata.jsonl"),
        max_num_frames=args.num_frames,
        frame_interval=1,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        is_i2v=args.image_encoder_path is not None,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )
    model = LightningModelForDataProcess(
        text_encoder_path=args.text_encoder_path,
        image_encoder_path=args.image_encoder_path,
        vae_path=args.vae_path,
        tiled=args.tiled,
        tile_size=(args.tile_size_height, args.tile_size_width),
        tile_stride=(args.tile_stride_height, args.tile_stride_width),
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto" if args.num_gpus is None else args.num_gpus,
        num_nodes=args.num_nodes,
        precision="bf16",
        strategy=args.training_strategy,
        default_root_dir=args.output_path,
    )
    trainer.test(model, dataloader)


def train(args):
    """Training pipeline."""
    dataset = TensorDataset(
        args.dataset_path,
        os.path.join(args.dataset_path, "metadata.jsonl"),
        steps_per_epoch=args.steps_per_epoch,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        # num_workers=args.dataloader_num_workers,
        num_workers=1,
        pin_memory=True,
    )
    model = LightningModelForTrain(
        dit_path=args.dit_path,
        learning_rate=args.learning_rate,
        train_architecture=args.train_architecture,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        init_lora_weights=args.init_lora_weights,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        pretrained_lora_path=args.pretrained_lora_path,
    )

    # Configure logging
    if args.use_swanlab:
        from swanlab.integration.pytorch_lightning import SwanLabLogger

        swanlab_config = {"UPPERFRAMEWORK": "DiffSynth-Studio"}
        swanlab_config.update(vars(args))
        swanlab_logger = SwanLabLogger(
            project="wan",
            name="wan",
            config=swanlab_config,
            mode=args.swanlab_mode,
            logdir=os.path.join(args.output_path, "swanlog"),
        )
        logger = [swanlab_logger]
    else:
        logger = None

    # Configure callbacks
    callbacks = []

    # Model checkpoint callback
    checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(
        dirpath=os.path.join(args.output_path, "checkpoints"),
        filename="wan-{epoch:02d}-{train_loss:.4f}",
        save_top_k=-1,
        every_n_train_steps=args.checkpoint_every_n_steps,
        save_on_train_epoch_end=True,
    )
    callbacks.append(checkpoint_callback)

    # Create a trainer
    # from lightning.pytorch.strategies import DeepSpeedStrategy
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto" if args.num_gpus is None else args.num_gpus,
        num_nodes=args.num_nodes,
        precision="bf16",
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks,
        logger=logger,
    )

    trainer.fit(model, dataloader)


if __name__ == "__main__":
    args = parse_args()

    if args.task == "data_process":
        data_process(args)
    elif args.task == "train":
        train(args)