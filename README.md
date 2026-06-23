# Project 2: Music Timbre Transfer

This repository contains my Project 2 implementation for **music timbre transfer** using spectrogram-based audio representation and LoRA fine-tuning on a Latent Diffusion Model (LDM).

The goal of this project is to transform the timbre of an input audio signal while preserving its musical content as much as possible. In this work, audio is represented as spectrogram images, and timbre adaptation is performed through LoRA fine-tuning.

## Demo

A demo page is available here:

https://votaquangnhat.github.io/music-timbre-transfer/demo/demo.html

## Data

The spectrogram dataset used in this project can be found here:

https://drive.google.com/drive/folders/16CNNIblK52s0WJYQJxwRKRBP4zGpjGPK

## LoRA Checkpoints

The trained LoRA checkpoints can be found here:

https://drive.google.com/drive/folders/1McAPt_xfDDOLpGfpglfqgDmJ5ZnwAROk

## External Pretrained Model

For external evaluation, this project also refers to the pretrained model from the paper:

**“Combining Audio Control and Style Transfer Using Latent Diffusion”**

The official implementation and pretrained model are available here:

https://github.com/NilsDem/control-transfer-diffusion/

## Overview

This project explores a diffusion-based approach to music timbre transfer. The general pipeline is:

1. Convert audio into spectrogram representations.
2. Fine-tune a latent diffusion model using LoRA.
3. Generate transferred spectrograms conditioned on the target timbre.
4. Convert the generated spectrograms back into audio.

## Notes

This repository is part of my university Project 2 work. The implementation is experimental and mainly focuses on exploring LoRA-based timbre adaptation with diffusion models.
