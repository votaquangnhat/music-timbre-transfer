from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from transformers import CLIPTextModel, CLIPTokenizer


class DiffusionLoRAModule(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        rank: int = 8,
        lora_alpha: Optional[int] = None,
        lora_dropout: float = 0.0,
        revision: Optional[str] = None,
        variant: Optional[str] = None,
        prompt_template: str = "a mel spectrogram of {timbre} music",
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.prompt_template = prompt_template

        self.tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=revision,
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=revision,
            variant=variant,
        )
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="vae",
            revision=revision,
            variant=variant,
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="unet",
            revision=revision,
            variant=variant,
        )
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="scheduler",
        )

        # Freeze base model. Only LoRA parameters will train.
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)

        if gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

        if lora_alpha is None:
            lora_alpha = rank

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=lora_alpha,
            init_lora_weights="gaussian",
            lora_dropout=lora_dropout,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        )
        self.unet.add_adapter(lora_config)

    @property
    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [p for p in self.unet.parameters() if p.requires_grad]

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters)

    def move_frozen_modules_to_device(self, device: torch.device, weight_dtype: torch.dtype) -> None:
        self.vae.to(device, dtype=weight_dtype)
        self.text_encoder.to(device, dtype=weight_dtype)

    def tokenize_timbres(self, timbres: list[str], device: torch.device) -> torch.Tensor:
        prompts = [self.prompt_template.format(timbre=timbre) for timbre in timbres]
        tokenized = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        return tokenized.input_ids.to(device)

    def training_loss(self, batch: dict, weight_dtype: torch.dtype) -> torch.Tensor:
        # take spectrograms -> encode -> add noise -> predict noise -> compute mse loss

        pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
        device = pixel_values.device

        latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        batch_size = latents.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=device,
        ).long()

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        input_ids = self.tokenize_timbres(batch["timbres"], device=device)
        encoder_hidden_states = self.text_encoder(input_ids)[0]

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unsupported prediction_type: {self.noise_scheduler.config.prediction_type}")

        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return F.mse_loss(model_pred.float(), target.float(), reduction="mean")

    @torch.no_grad()
    def validation_loss(self, batch: dict, weight_dtype: torch.dtype) -> torch.Tensor:
        return self.training_loss(batch, weight_dtype=weight_dtype)

    def save_lora(self, output_dir: str, accelerator: Accelerator) -> None:
        """Save only LoRA weights, not the full base model."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        unwrapped_unet = accelerator.unwrap_model(self.unet)
        unet_lora_state_dict = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(unwrapped_unet)
        )

        StableDiffusionPipeline.save_lora_weights(
            save_directory=str(output_path),
            unet_lora_layers=unet_lora_state_dict,
            safe_serialization=True,
        )
