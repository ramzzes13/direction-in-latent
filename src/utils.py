"""Utility functions for the semantic direction pipeline."""

import gc
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def get_best_gpu() -> str:
    """Find GPU with most free memory. Returns device string like 'cuda:3'."""
    if not torch.cuda.is_available():
        return "cpu"
    best_device = 0
    best_free = 0
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        if free > best_free:
            best_free = free
            best_device = i
    print(f"Selected GPU {best_device} with {best_free / 1e9:.1f} GB free")
    return f"cuda:{best_device}"


def save_checkpoint(data: dict, path: str):
    """Save a stage checkpoint (torch tensors + metadata)."""
    ensure_dir(os.path.dirname(path))
    torch.save(data, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(path: str) -> dict:
    """Load a stage checkpoint."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    print(f"Checkpoint loaded from {path}")
    return data


def save_json(data: Any, path: str):
    """Save data as JSON."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> Any:
    """Load data from JSON."""
    with open(path, "r") as f:
        return json.load(f)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert (C, H, W) tensor in [0,1] to PIL Image."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    arr = (tensor.clamp(0, 1).cpu().float().numpy() * 255).astype(np.uint8)
    arr = arr.transpose(1, 2, 0)  # CHW -> HWC
    return Image.fromarray(arr)


def save_image(tensor: torch.Tensor, path: str):
    """Save tensor as PNG image."""
    ensure_dir(os.path.dirname(path))
    img = tensor_to_pil(tensor)
    img.save(path)


def free_gpu_memory():
    """Force garbage collection and clear GPU cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
