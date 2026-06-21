import csv
import os
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

from lora_module import DiffusionLoRAModule
from train_config import parse_args
from utils import (
    compute_validation_loss,
    get_weight_dtype,
    make_dataloader,
    maybe_make_validation_dataloader,
    resolve_training_steps,
    rotate_checkpoints,
    save_training_config,
)

logger = get_logger(__name__)

def init_metrics_file(output_dir: str) -> None:
    metrics_path = Path(output_dir) / "metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "train_loss", "val_loss", "lr"])
        writer.writeheader()


def append_metrics(
    output_dir: str,
    step: int,
    train_loss: float | None,
    val_loss: float | None,
    lr: float,
) -> None:
    metrics_path = Path(output_dir) / "metrics.csv"
    with open(metrics_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "train_loss", "val_loss", "lr"])
        writer.writerow({
            "step": step,
            "train_loss": "" if train_loss is None else train_loss,
            "val_loss": "" if val_loss is None else val_loss,
            "lr": lr,
        })

def main() -> None:
    # 1. config, acce, module
    config = parse_args()

    project_config = ProjectConfiguration(
        project_dir=config.output_dir,
        logging_dir=os.path.join(config.output_dir, config.logging_dir),
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=None if config.mixed_precision == "no" else config.mixed_precision,
        project_config=project_config,
    )

    if config.seed is not None:
        set_seed(config.seed)

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if accelerator.is_main_process:
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        save_training_config(config)
        init_metrics_file(config.output_dir)

    module = DiffusionLoRAModule(
        pretrained_model_name_or_path=config.pretrained_model_name_or_path,
        rank=config.rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        revision=config.revision,
        variant=config.variant,
        prompt_template=config.prompt_template,
        gradient_checkpointing=config.gradient_checkpointing,
    )

    if config.scale_lr:
        config.learning_rate = (
            config.learning_rate
            * config.gradient_accumulation_steps
            * config.train_batch_size
            * accelerator.num_processes
        )

    # 2. dataloader, optimizer
    train_dataloader = make_dataloader(
        config=config,
        subset=config.train_subset,
        max_samples=config.max_train_samples,
        shuffle=True,
    )
    val_dataloader = maybe_make_validation_dataloader(config, logger)

    optimizer = torch.optim.AdamW(
        module.trainable_parameters, # only lora params
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
        eps=config.adam_epsilon,
    )

    max_train_steps, num_train_epochs = resolve_training_steps(config, train_dataloader)

    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )

    module.unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        module.unet,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )
    trainable_params_for_clip = [p for p in module.unet.parameters() if p.requires_grad]

    if val_dataloader is not None:
        val_dataloader = accelerator.prepare(val_dataloader)

    weight_dtype = get_weight_dtype(accelerator)
    module.move_frozen_modules_to_device(accelerator.device, weight_dtype)

    logger.info(f"Training timbre: {config.timbre}")
    logger.info(f"Spectrogram dir: {config.spectrograms_dir}")
    logger.info(f"Train batches: {len(train_dataloader)}")
    logger.info(f"Batch size/device: {config.train_batch_size}")
    logger.info(f"Gradient accumulation: {config.gradient_accumulation_steps}")
    logger.info(f"Max train steps: {max_train_steps}")
    logger.info(f"Trainable LoRA params: {module.num_trainable_parameters():,}")

    global_step = 0
    progress_bar = tqdm(
        range(max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="Training",
    )

    # 3. loop

    module.unet.train()
    for epoch in range(num_train_epochs):
        for batch in train_dataloader:
            with accelerator.accumulate(module.unet):
                loss = module.training_loss(batch, weight_dtype=weight_dtype)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params_for_clip, config.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1

                train_loss_value = accelerator.gather(loss.detach().reshape(1)).mean().item()
                lr_value = lr_scheduler.get_last_lr()[0]
                val_loss_value = None

                progress_bar.update(1)
                progress_bar.set_postfix(
                    loss=f"{train_loss_value:.4f}",
                    lr=f"{lr_value:.2e}",
                )

                if accelerator.is_main_process and config.save_steps > 0 and global_step % config.save_steps == 0:
                    checkpoint_dir = Path(config.output_dir) / f"checkpoint-{global_step}"
                    module.save_lora(str(checkpoint_dir), accelerator)
                    rotate_checkpoints(config.output_dir, config.checkpoint_limit)
                    logger.info(f"Saved checkpoint: {checkpoint_dir}")

                if (
                    val_dataloader is not None
                    and config.validation_steps > 0
                    and global_step % config.validation_steps == 0
                ):
                    val_loss_value = compute_validation_loss(
                        module=module,
                        val_dataloader=val_dataloader,
                        accelerator=accelerator,
                        weight_dtype=weight_dtype,
                        max_batches=config.validation_max_batches,
                    )
                    logger.info(f"step={global_step} validation_loss={val_loss_value:.6f}")

                if accelerator.is_main_process:
                    append_metrics(
                        output_dir=config.output_dir,
                        step=global_step,
                        train_loss=train_loss_value,
                        val_loss=val_loss_value,
                        lr=lr_value,
                    )

            if global_step >= max_train_steps:
                break

        if global_step >= max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        module.save_lora(config.output_dir, accelerator)
        logger.info(f"Saved final LoRA weights to: {config.output_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()