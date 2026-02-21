"""Stage 2: Edit generation using h-space directions."""

import os
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

from src.config import PipelineConfig
from src.hooks import BottleneckHook
from src.sampler import ddim_sample
from src.utils import save_image, save_json, ensure_dir


class EditGenerator:
    """
    Generates positive/negative edit pairs for each PCA direction.

    For direction v_k with strength alpha:
    - Original: standard generation from seed s
    - Positive: generation with delta_h = +alpha * v_k at each timestep
    - Negative: generation with delta_h = -alpha * v_k at each timestep
    """

    def __init__(self, unet, scheduler, config: PipelineConfig):
        self.unet = unet
        self.scheduler = scheduler
        self.config = config
        self.device = str(next(unet.parameters()).device)
        self.dtype = next(unet.parameters()).dtype

    def generate_edit_pairs(self, pca_data: Dict) -> Dict:
        """
        Generate all edit pairs and save to disk.

        Args:
            pca_data: Output from Stage 1 (contains 'components' tensor)

        Returns:
            Metadata dict with paths and generation parameters.
        """
        edit_cfg = self.config.edit
        samp_cfg = self.config.sampler
        components = pca_data["components"]  # (K, T, C, H, W)

        num_dirs = min(edit_cfg.num_directions, components.shape[0])
        seeds = edit_cfg.edit_seeds[:edit_cfg.num_seeds_per_direction]
        alpha = edit_cfg.default_alpha

        edits_dir = os.path.join(self.config.output_dir, "stage2", "edits")
        metadata = {
            "num_directions": num_dirs,
            "alpha": alpha,
            "num_seeds_per_direction": len(seeds),
            "num_inference_steps": samp_cfg.num_inference_steps,
            "eta": samp_cfg.eta,
            "edits": [],
        }

        total = num_dirs * len(seeds)
        pbar = tqdm(total=total, desc="Generating edit pairs")

        for k in range(num_dirs):
            direction = components[k]  # (T, C, H, W)
            pc_dir = os.path.join(edits_dir, f"pc{k:02d}")
            ensure_dir(pc_dir)

            for seed in seeds:
                orig, pos, neg = self._generate_single_edit(
                    seed=seed,
                    direction=direction,
                    alpha=alpha,
                    num_steps=samp_cfg.num_inference_steps,
                    eta=samp_cfg.eta,
                )

                # Save images
                orig_path = os.path.join(pc_dir, f"seed_{seed:04d}_orig.png")
                pos_path = os.path.join(pc_dir, f"seed_{seed:04d}_pos.png")
                neg_path = os.path.join(pc_dir, f"seed_{seed:04d}_neg.png")

                save_image(orig, orig_path)
                save_image(pos, pos_path)
                save_image(neg, neg_path)

                metadata["edits"].append({
                    "direction_idx": k,
                    "seed": seed,
                    "orig_path": os.path.relpath(orig_path, self.config.output_dir),
                    "pos_path": os.path.relpath(pos_path, self.config.output_dir),
                    "neg_path": os.path.relpath(neg_path, self.config.output_dir),
                })

                pbar.update(1)

        pbar.close()

        # Save metadata
        meta_path = os.path.join(self.config.output_dir, "stage2", "edit_metadata.json")
        save_json(metadata, meta_path)
        print(f"Edit metadata saved to {meta_path}")

        return metadata

    def _generate_single_edit(
        self,
        seed: int,
        direction: torch.Tensor,
        alpha: float,
        num_steps: int,
        eta: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate (original, positive_edit, negative_edit) for one seed+direction.

        All three generations use the same noise sequence via identical generator seeds.
        """
        # 1. Generate original (no edit)
        hook_orig = BottleneckHook(self.unet, mode="capture")
        with hook_orig:
            gen = torch.Generator(device=self.device).manual_seed(seed)
            img_orig, _ = ddim_sample(
                self.unet, self.scheduler, hook_orig,
                num_inference_steps=num_steps, eta=eta,
                generator=gen, device=self.device, dtype=self.dtype,
            )

        # 2. Generate positive edit (+alpha * direction)
        hook_pos = BottleneckHook(self.unet, mode="edit")
        hook_pos.set_direction(direction, alpha=alpha, num_timesteps=num_steps)
        with hook_pos:
            gen = torch.Generator(device=self.device).manual_seed(seed)
            img_pos, _ = ddim_sample(
                self.unet, self.scheduler, hook_pos,
                num_inference_steps=num_steps, eta=eta,
                generator=gen, device=self.device, dtype=self.dtype,
            )

        # 3. Generate negative edit (-alpha * direction)
        hook_neg = BottleneckHook(self.unet, mode="edit")
        hook_neg.set_direction(direction, alpha=-alpha, num_timesteps=num_steps)
        with hook_neg:
            gen = torch.Generator(device=self.device).manual_seed(seed)
            img_neg, _ = ddim_sample(
                self.unet, self.scheduler, hook_neg,
                num_inference_steps=num_steps, eta=eta,
                generator=gen, device=self.device, dtype=self.dtype,
            )

        return img_orig, img_pos, img_neg
