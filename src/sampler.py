"""DDIM sampling loop with h-space hook support."""

from typing import Optional, Tuple

import torch
from diffusers import UNet2DModel, DDIMScheduler

from src.hooks import BottleneckHook


@torch.no_grad()
def ddim_sample(
    unet: UNet2DModel,
    scheduler: DDIMScheduler,
    hook: BottleneckHook,
    num_inference_steps: int = 50,
    eta: float = 1.0,
    generator: Optional[torch.Generator] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Run DDIM sampling with bottleneck hook active.

    Args:
        unet: The U-Net model
        scheduler: DDIM scheduler
        hook: BottleneckHook instance (must already be registered)
        num_inference_steps: Number of denoising steps
        eta: Noise factor (0=deterministic DDIM, 1=stochastic DDPM)
        generator: Torch generator for reproducibility
        device: Device string
        dtype: Tensor dtype

    Returns:
        image: (1, 3, 256, 256) generated image tensor in [0, 1]
        h_activations: (T, C, H, W) bottleneck activations if capturing, else None
    """
    scheduler.set_timesteps(num_inference_steps, device=device)

    # Initialize random noise x_T
    x_t = torch.randn(
        1, 3, 256, 256, generator=generator, device=device, dtype=dtype
    )

    hook.reset()

    for t in scheduler.timesteps:
        noise_pred = unet(x_t, t).sample
        step_output = scheduler.step(
            noise_pred, t, x_t, eta=eta, generator=generator
        )
        x_t = step_output.prev_sample

    # Convert from [-1, 1] to [0, 1]
    image = (x_t / 2 + 0.5).clamp(0, 1)

    h_activations = None
    if hook.mode in ("capture", "capture_and_edit"):
        h_activations = hook.get_activations()

    return image, h_activations


def generate_with_seed(
    unet: UNet2DModel,
    scheduler: DDIMScheduler,
    hook: BottleneckHook,
    seed: int,
    num_inference_steps: int = 50,
    eta: float = 1.0,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Deterministic generation from a specific seed."""
    generator = torch.Generator(device=device).manual_seed(seed)
    return ddim_sample(
        unet, scheduler, hook,
        num_inference_steps=num_inference_steps,
        eta=eta,
        generator=generator,
        device=device,
        dtype=dtype,
    )
