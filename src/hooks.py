"""U-Net bottleneck hook management for capturing and editing h-space activations."""

from typing import List, Optional

import torch
from diffusers import UNet2DModel


class BottleneckHook:
    """
    Manages forward hooks on UNet2DModel.mid_block to capture/edit h-space.

    Modes:
    - 'capture': Records h_t at each timestep during generation
    - 'edit': Adds delta_h to h_t at each timestep during generation
    - 'capture_and_edit': Both captures original and injects delta
    """

    def __init__(self, unet: UNet2DModel, mode: str = "capture"):
        assert mode in ("capture", "edit", "capture_and_edit")
        self.unet = unet
        self.mode = mode
        self.captured_h: List[torch.Tensor] = []
        self.delta_h: Optional[List[torch.Tensor]] = None
        self.timestep_idx: int = 0
        self._hook_handle = None

    def _hook_fn(self, module, input, output):
        """
        Forward hook on unet.mid_block.

        The mid_block output is the bottleneck activation h_t with
        shape (batch, 512, 8, 8) for google/ddpm-celebahq-256.
        """
        if self.mode in ("capture", "capture_and_edit"):
            self.captured_h.append(output.detach().cpu().float())

        if self.mode in ("edit", "capture_and_edit"):
            if self.delta_h is not None and self.timestep_idx < len(self.delta_h):
                delta = self.delta_h[self.timestep_idx].to(
                    device=output.device, dtype=output.dtype
                )
                output = output + delta

        self.timestep_idx += 1
        return output

    def register(self):
        """Register the forward hook on unet.mid_block."""
        self._hook_handle = self.unet.mid_block.register_forward_hook(self._hook_fn)
        return self

    def remove(self):
        """Remove the hook and clean up."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def reset(self):
        """Reset captured data between samples."""
        self.captured_h = []
        self.timestep_idx = 0

    def get_activations(self) -> torch.Tensor:
        """Return captured activations as (T, C, H, W) tensor."""
        if not self.captured_h:
            raise RuntimeError("No activations captured. Did you run generation?")
        return torch.stack(self.captured_h, dim=0).squeeze(1)  # (T, C, H, W)

    def set_direction(self, direction: torch.Tensor, alpha: float, num_timesteps: int):
        """
        Set the editing direction.

        Args:
            direction: (T, C, H, W) per-timestep PCA component
            alpha: scaling factor (positive for pos edit, negative for neg edit)
            num_timesteps: number of DDIM timesteps
        """
        if direction.dim() == 3:
            # Single direction broadcast across all timesteps
            self.delta_h = [alpha * direction.unsqueeze(0)] * num_timesteps
        elif direction.dim() == 4:
            # Per-timestep direction: (T, C, H, W)
            assert direction.shape[0] == num_timesteps, (
                f"Direction has {direction.shape[0]} timesteps but sampler uses {num_timesteps}"
            )
            self.delta_h = [alpha * direction[t].unsqueeze(0) for t in range(num_timesteps)]
        else:
            raise ValueError(f"Direction must be 3D or 4D, got {direction.dim()}D")

    def __enter__(self):
        self.register()
        return self

    def __exit__(self, *args):
        self.remove()
