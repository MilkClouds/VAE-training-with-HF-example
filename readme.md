# VAE-training-with-HF-example

This repository contains the example code for simple VAE training & visualization script using Huggingface libraries, such as Accelerate, Datasets, Diffusers.

Basically, the training is runnable on RTX 3080 with 10GB VRAM, with bf16 precision, consuming ~12 minutes.
Trainable params: 3.4M

# Setup

```bash
poetry install
```

# How to run

```bash
# config your accelerate
poetry run accelerate config
# launch training
poetry run accelerate launch train.py
```