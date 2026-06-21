import json
import math
import shutil
from pathlib import Path
from typing import Optional

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from dataset import TimbreSpectrogramDataset, collate_fn
from train_config import TrainConfig
from lora_module import DiffusionLoRAModule


def get_weight_dtype(accelerator: Accelerator) -> torch.dtype:
    if accelerator.mixed_precision == "fp16":
        return torch.float16
    if accelerator.mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def save_training_config(config: TrainConfig) -> None:
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "training_args.json", "w", encoding="utf-8") as f:
        json.dump(config.__dict__, f, indent=2)


def make_dataloader(
    *,
    config: TrainConfig,
    subset: str,
    max_samples: Optional[int],
    shuffle: bool,
) -> DataLoader:
    dataset = TimbreSpectrogramDataset(
        spectrograms_dir=config.spectrograms_dir,
        timbre=config.timbre,
        subset=subset,
        image_size=config.image_size,
        max_samples=max_samples,
        seed=config.seed,
        verify_files=config.verify_files,
    )
    return DataLoader(
        dataset,
        shuffle=shuffle,
        collate_fn=collate_fn,
        batch_size=config.train_batch_size,
        num_workers=config.dataloader_num_workers,
        pin_memory=True,
    )


def maybe_make_validation_dataloader(config: TrainConfig, logger) -> Optional[DataLoader]:
    try:
        return make_dataloader(
            config=config,
            subset=config.validation_subset,
            max_samples=config.max_validation_samples,
            shuffle=False,
        )
    except Exception as exc:
        logger.warning(f"Validation dataloader disabled: {exc}")
        return None


def rotate_checkpoints(output_dir: str, limit: Optional[int]) -> None:
    if limit is None or limit <= 0:
        return

    output_path = Path(output_dir)
    checkpoints = sorted(
        [p for p in output_path.glob("checkpoint-*") if p.is_dir()],
        key=lambda p: int(p.name.split("-")[-1]),
    )
    excess = len(checkpoints) - limit
    if excess <= 0:
        return

    for ckpt in checkpoints[:excess]:
        shutil.rmtree(ckpt, ignore_errors=True)


def resolve_training_steps(config: TrainConfig, train_dataloader: DataLoader) -> tuple[int, int]:
    steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    if config.max_train_steps is None or config.max_train_steps <= 0:
        max_train_steps = config.num_train_epochs * steps_per_epoch
        num_train_epochs = config.num_train_epochs
    else:
        max_train_steps = config.max_train_steps
        num_train_epochs = math.ceil(max_train_steps / steps_per_epoch)
    return max_train_steps, num_train_epochs


@torch.no_grad()
def compute_validation_loss(
    *,
    module: DiffusionLoRAModule,
    val_dataloader: DataLoader,
    accelerator: Accelerator,
    weight_dtype: torch.dtype,
    max_batches: int = 8,
) -> float:
    module.unet.eval()
    losses = []

    for step, batch in enumerate(val_dataloader):
        if step >= max_batches:
            break
        loss = module.validation_loss(batch, weight_dtype=weight_dtype)
        losses.append(accelerator.gather(loss.detach()).mean().item())

    module.unet.train()
    return float(sum(losses) / max(1, len(losses)))
