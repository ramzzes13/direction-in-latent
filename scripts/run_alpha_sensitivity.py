"""Experiment 4: Alpha sensitivity analysis.

Tests how edit strength (alpha) affects CLIP labeling confidence.
For each alpha value, generates edits and measures CLIP delta scores.
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from PIL import Image

from src.config import load_config, save_config
from src.model import load_ddpm_model
from src.hooks import BottleneckHook
from src.sampler import ddim_sample
from src.labeling import CLIPLabeler
from src.utils import (
    load_checkpoint, save_json, save_image, ensure_dir,
    free_gpu_memory, get_best_gpu,
)
from src.visualization import create_edit_grid


def generate_edit_for_alpha(unet, scheduler, direction, seed, alpha, num_steps, eta, device, dtype):
    """Generate (orig, pos, neg) triplet for a given alpha."""
    # Original (no edit)
    hook_orig = BottleneckHook(unet, mode="capture")
    with hook_orig:
        gen = torch.Generator(device=device).manual_seed(seed)
        img_orig, _ = ddim_sample(
            unet, scheduler, hook_orig,
            num_inference_steps=num_steps, eta=eta,
            generator=gen, device=device, dtype=dtype,
        )

    # Positive edit
    hook_pos = BottleneckHook(unet, mode="edit")
    hook_pos.set_direction(direction, alpha=alpha, num_timesteps=num_steps)
    with hook_pos:
        gen = torch.Generator(device=device).manual_seed(seed)
        img_pos, _ = ddim_sample(
            unet, scheduler, hook_pos,
            num_inference_steps=num_steps, eta=eta,
            generator=gen, device=device, dtype=dtype,
        )

    # Negative edit
    hook_neg = BottleneckHook(unet, mode="edit")
    hook_neg.set_direction(direction, alpha=-alpha, num_timesteps=num_steps)
    with hook_neg:
        gen = torch.Generator(device=device).manual_seed(seed)
        img_neg, _ = ddim_sample(
            unet, scheduler, hook_neg,
            num_inference_steps=num_steps, eta=eta,
            generator=gen, device=device, dtype=dtype,
        )

    return img_orig, img_pos, img_neg


def main():
    parser = argparse.ArgumentParser(description="Experiment 4: Alpha Sensitivity")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    parser.add_argument("--num-directions", type=int, default=5,
                        help="Number of top PCA directions to test")
    parser.add_argument("--num-seeds", type=int, default=3,
                        help="Seeds per direction per alpha")
    args = parser.parse_args()

    config = load_config(args.config)
    alpha_values = args.alphas
    num_dirs = args.num_directions
    seeds = config.edit.edit_seeds[:args.num_seeds]

    device = config.model.device
    if device == "auto":
        device = get_best_gpu()
        config.model.device = device

    exp_dir = os.path.join(config.output_dir, "experiment4_alpha")
    ensure_dir(exp_dir)

    # Load PCA checkpoint
    pca_path = os.path.join(config.output_dir, "stage1", "pca_components.pt")
    pca_data = load_checkpoint(pca_path)
    components = pca_data["components"]  # (K, T, C, H, W)

    # Load DDPM model
    print("Loading DDPM model...")
    unet, scheduler = load_ddpm_model(config.model)
    dtype = next(unet.parameters()).dtype

    num_steps = config.sampler.num_inference_steps
    eta = config.sampler.eta

    # Generate edits at each alpha
    all_edits = {}  # alpha -> list of edit dicts

    total = len(alpha_values) * num_dirs * len(seeds)
    done = 0

    for alpha in alpha_values:
        print(f"\n--- Alpha = {alpha} ---")
        alpha_dir = os.path.join(exp_dir, f"alpha_{alpha:.1f}")
        edits = []

        for k in range(num_dirs):
            direction = components[k]  # (T, C, H, W)
            pc_dir = os.path.join(alpha_dir, f"pc{k:02d}")
            ensure_dir(pc_dir)

            for seed in seeds:
                img_orig, img_pos, img_neg = generate_edit_for_alpha(
                    unet, scheduler, direction, seed, alpha,
                    num_steps, eta, device, dtype,
                )

                orig_path = os.path.join(pc_dir, f"seed_{seed:04d}_orig.png")
                pos_path = os.path.join(pc_dir, f"seed_{seed:04d}_pos.png")
                neg_path = os.path.join(pc_dir, f"seed_{seed:04d}_neg.png")

                save_image(img_orig, orig_path)
                save_image(img_pos, pos_path)
                save_image(img_neg, neg_path)

                edits.append({
                    "direction_idx": k,
                    "seed": seed,
                    "pos_path": os.path.relpath(pos_path, config.output_dir),
                    "neg_path": os.path.relpath(neg_path, config.output_dir),
                    "orig_path": os.path.relpath(orig_path, config.output_dir),
                })

                done += 1
                print(f"  [{done}/{total}] PC{k} seed={seed} alpha={alpha}")

        all_edits[alpha] = {
            "alpha": alpha,
            "num_directions": num_dirs,
            "num_seeds": len(seeds),
            "edits": edits,
        }

    # Free DDPM model
    del unet, scheduler
    free_gpu_memory()

    # Run CLIP labeling for each alpha
    print("\nLoading CLIP model for scoring...")
    clip_labeler = CLIPLabeler(config.labeling, device)
    clip_labeler.load_model()

    results = {}
    for alpha in alpha_values:
        print(f"\n--- CLIP scoring for alpha = {alpha} ---")
        clip_scores = clip_labeler.label_directions(
            all_edits[alpha], config.output_dir
        )
        results[str(alpha)] = {
            "clip_scores": clip_scores,
            "alpha": alpha,
        }

    clip_labeler.unload_model()

    # Save all results
    save_json(results, os.path.join(exp_dir, "alpha_sensitivity_results.json"))

    # Create summary and plot
    _create_alpha_plot(results, alpha_values, num_dirs, exp_dir)
    _print_summary(results, alpha_values, num_dirs)

    # Create grids for visual comparison at each alpha
    for alpha in alpha_values:
        alpha_dir = os.path.join(exp_dir, f"alpha_{alpha:.1f}")
        grids_dir = os.path.join(alpha_dir, "grids")
        ensure_dir(grids_dir)

        for k in range(num_dirs):
            edits_k = [e for e in all_edits[alpha]["edits"]
                       if e["direction_idx"] == k]
            images = []
            for e in edits_k:
                neg = Image.open(os.path.join(config.output_dir, e["neg_path"]))
                orig = Image.open(os.path.join(config.output_dir, e["orig_path"]))
                pos = Image.open(os.path.join(config.output_dir, e["pos_path"]))
                images.append((neg, orig, pos))

            label = results[str(alpha)]["clip_scores"].get(
                f"direction_{k}", {}
            ).get("top_label", "")

            create_edit_grid(
                direction_idx=k, images=images, alpha=alpha,
                label=label,
                save_path=os.path.join(grids_dir, f"pc{k:02d}_grid.png"),
            )

    save_json(
        {str(a): all_edits[a] for a in alpha_values},
        os.path.join(exp_dir, "all_edit_metadata.json"),
    )

    print(f"\nExperiment 4 complete. Results in {exp_dir}/")


def _create_alpha_plot(results, alpha_values, num_dirs, exp_dir):
    """Create alpha sensitivity plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Top CLIP score vs alpha for each direction
    for k in range(num_dirs):
        scores = []
        for alpha in alpha_values:
            key = f"direction_{k}"
            score = results[str(alpha)]["clip_scores"].get(key, {}).get("top_score", 0)
            scores.append(score)
        axes[0].plot(alpha_values, scores, "o-", label=f"PC{k}")

    axes[0].set_xlabel("Alpha (edit strength)")
    axes[0].set_ylabel("Top CLIP Delta Score")
    axes[0].set_title("CLIP Confidence vs Edit Strength")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Mean top score across directions vs alpha
    mean_scores = []
    for alpha in alpha_values:
        dir_scores = []
        for k in range(num_dirs):
            key = f"direction_{k}"
            s = results[str(alpha)]["clip_scores"].get(key, {}).get("top_score", 0)
            dir_scores.append(abs(s))
        mean_scores.append(np.mean(dir_scores))

    axes[1].plot(alpha_values, mean_scores, "s-", color="steelblue", linewidth=2)
    axes[1].set_xlabel("Alpha (edit strength)")
    axes[1].set_ylabel("Mean |Top CLIP Delta Score|")
    axes[1].set_title("Average CLIP Confidence vs Edit Strength")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(exp_dir, "alpha_sensitivity_plot.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Alpha sensitivity plot saved to {exp_dir}/alpha_sensitivity_plot.png")


def _print_summary(results, alpha_values, num_dirs):
    """Print summary table."""
    print("\n=== Alpha Sensitivity Summary ===")
    print(f"{'Alpha':>8} | ", end="")
    for k in range(num_dirs):
        print(f"{'PC'+str(k):>25} | ", end="")
    print()
    print("-" * (10 + 28 * num_dirs))

    for alpha in alpha_values:
        print(f"{alpha:8.1f} | ", end="")
        for k in range(num_dirs):
            key = f"direction_{k}"
            entry = results[str(alpha)]["clip_scores"].get(key, {})
            label = entry.get("top_label", "?")
            score = entry.get("top_score", 0)
            short = label[:18]
            print(f"{short:>18} ({score:+.4f}) | ", end="")
        print()


if __name__ == "__main__":
    main()
