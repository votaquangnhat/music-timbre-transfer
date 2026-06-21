"""Argument parsing for timbre LoRA training."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainConfig:
    # Model/data
    pretrained_model_name_or_path: str = "riffusion/riffusion-model-v1"
    revision: Optional[str] = None
    variant: Optional[str] = None
    spectrograms_dir: str = "data/spectrogram"
    timbre: str = "piano"
    train_subset: str = "train"
    validation_subset: str = "validation"
    prompt_template: str = "a mel spectrogram of {timbre} music"
    image_size: int = 512
    max_train_samples: Optional[int] = None
    max_validation_samples: Optional[int] = 64
    verify_files: bool = False

    # Output/training
    output_dir: str = "outputs/lora_piano"
    logging_dir: str = "logs"
    seed: int = 42
    train_batch_size: int = 1
    num_train_epochs: int = 100
    max_train_steps: int = 1000
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = False
    learning_rate: float = 1e-4
    scale_lr: bool = False
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    mixed_precision: str = "fp16"
    allow_tf32: bool = False
    dataloader_num_workers: int = 2

    # LoRA
    rank: int = 8
    lora_alpha: Optional[int] = None
    lora_dropout: float = 0.0

    # Validation/checkpointing
    validation_steps: int = 100
    validation_max_batches: int = 8
    save_steps: int = 500
    checkpoint_limit: int = 3


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train one unpaired LoRA adapter for one music timbre.")

    # Model/data
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=TrainConfig.pretrained_model_name_or_path)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--spectrograms_dir", type=str, default=TrainConfig.spectrograms_dir)
    parser.add_argument("--timbre", type=str, required=True, help="Example: piano or flute")
    parser.add_argument("--train_subset", type=str, default=TrainConfig.train_subset)
    parser.add_argument("--validation_subset", type=str, default=TrainConfig.validation_subset)
    parser.add_argument("--prompt_template", type=str, default=TrainConfig.prompt_template)
    parser.add_argument("--image_size", type=int, default=TrainConfig.image_size)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_validation_samples", type=int, default=TrainConfig.max_validation_samples)
    parser.add_argument("--verify_files", action="store_true")

    # Output/training
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--logging_dir", type=str, default=TrainConfig.logging_dir)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--train_batch_size", type=int, default=TrainConfig.train_batch_size)
    parser.add_argument("--num_train_epochs", type=int, default=TrainConfig.num_train_epochs)
    parser.add_argument("--max_train_steps", type=int, default=TrainConfig.max_train_steps)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=TrainConfig.gradient_accumulation_steps)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--scale_lr", action="store_true")
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default=TrainConfig.lr_scheduler,
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=TrainConfig.lr_warmup_steps)
    parser.add_argument("--adam_beta1", type=float, default=TrainConfig.adam_beta1)
    parser.add_argument("--adam_beta2", type=float, default=TrainConfig.adam_beta2)
    parser.add_argument("--adam_weight_decay", type=float, default=TrainConfig.adam_weight_decay)
    parser.add_argument("--adam_epsilon", type=float, default=TrainConfig.adam_epsilon)
    parser.add_argument("--max_grad_norm", type=float, default=TrainConfig.max_grad_norm)
    parser.add_argument("--mixed_precision", type=str, default=TrainConfig.mixed_precision, choices=["no", "fp16", "bf16"])
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=TrainConfig.dataloader_num_workers)

    # LoRA
    parser.add_argument("--rank", type=int, default=TrainConfig.rank)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=TrainConfig.lora_dropout)

    # Validation/checkpointing
    parser.add_argument("--validation_steps", type=int, default=TrainConfig.validation_steps)
    parser.add_argument("--validation_max_batches", type=int, default=TrainConfig.validation_max_batches)
    parser.add_argument("--save_steps", type=int, default=TrainConfig.save_steps)
    parser.add_argument("--checkpoint_limit", type=int, default=TrainConfig.checkpoint_limit)

    args = parser.parse_args()
    config = TrainConfig(**vars(args))
    if config.lora_alpha is None:
        config.lora_alpha = config.rank
    return config
