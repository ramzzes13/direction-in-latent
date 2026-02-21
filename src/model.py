"""DDPM model loading and scheduler setup."""

from typing import Tuple

import torch
from diffusers import UNet2DModel, DDIMScheduler

from src.config import ModelConfig
from src.utils import get_best_gpu


def load_ddpm_model(config: ModelConfig) -> Tuple[UNet2DModel, DDIMScheduler]:
    """
    Load the pre-trained UNet2DModel and create a DDIMScheduler.

    Returns the model in fp16 on the specified device, plus the scheduler.
    """
    device = config.device
    if device == "auto":
        device = get_best_gpu()

    dtype = torch.float16 if config.torch_dtype == "float16" else torch.float32

    print(f"Loading UNet from {config.model_id}...")
    unet = UNet2DModel.from_pretrained(config.model_id)
    unet = unet.to(device=device, dtype=dtype)
    unet.eval()

    print(f"Loading DDIMScheduler from {config.model_id}...")
    scheduler = DDIMScheduler.from_pretrained(config.model_id)

    # Verify expected architecture
    mid_block = unet.mid_block
    print(f"UNet mid_block type: {type(mid_block).__name__}")
    print(f"Model loaded on {device} with dtype {dtype}")

    return unet, scheduler
