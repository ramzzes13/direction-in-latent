"""Visualization utilities for edit grids, label heatmaps, and PCA variance plots."""

import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.utils import ensure_dir


def create_edit_grid(
    direction_idx: int,
    images: List[Tuple[Image.Image, Image.Image, Image.Image]],
    alpha: float,
    label: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a grid visualization for one PCA direction.

    Args:
        direction_idx: PCA component index
        images: List of (neg, orig, pos) PIL Image tuples per seed
        alpha: Edit strength
        label: Optional text label for this direction
        save_path: Path to save PNG
    """
    n_seeds = len(images)
    fig, axes = plt.subplots(n_seeds, 3, figsize=(9, 3 * n_seeds))

    if n_seeds == 1:
        axes = axes[np.newaxis, :]

    title = f"PC {direction_idx}"
    if label:
        title += f": {label}"
    title += f" (alpha={alpha})"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    col_titles = [f"Negative (-{alpha})", "Original", f"Positive (+{alpha})"]
    for col, ct in enumerate(col_titles):
        axes[0, col].set_title(ct, fontsize=10)

    for row, (neg, orig, pos) in enumerate(images):
        for col, img in enumerate([neg, orig, pos]):
            axes[row, col].imshow(np.array(img))
            axes[row, col].axis("off")

    plt.tight_layout()

    if save_path:
        ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Grid saved to {save_path}")

    return fig


def create_label_summary_table(
    clip_scores: Dict,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a heatmap of CLIP scores for each direction x attribute.

    Args:
        clip_scores: Output from CLIPLabeler.label_directions()
        save_path: Path to save PNG
    """
    directions = sorted(clip_scores.keys())
    if not directions:
        return plt.figure()

    # Get attribute list from first direction
    first_dir = clip_scores[directions[0]]
    attributes = list(first_dir["scores"].keys())

    # Build matrix
    matrix = np.zeros((len(directions), len(attributes)))
    for i, d in enumerate(directions):
        for j, attr in enumerate(attributes):
            matrix[i, j] = clip_scores[d]["scores"].get(attr, 0.0)

    fig, ax = plt.subplots(figsize=(max(14, len(attributes) * 0.7), len(directions) * 0.6 + 2))

    im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-0.1, vmax=0.1)

    # Labels
    short_attrs = [a.replace("a person ", "").replace("a face ", "")
                   for a in attributes]
    ax.set_xticks(range(len(attributes)))
    ax.set_xticklabels(short_attrs, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(directions)))
    ax.set_yticklabels([d.replace("direction_", "PC") for d in directions], fontsize=9)

    # Highlight top label per direction
    for i, d in enumerate(directions):
        top_attr = clip_scores[d]["top_label"]
        j = attributes.index(top_attr)
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                   fill=False, edgecolor="black", linewidth=2))

    plt.colorbar(im, ax=ax, label="CLIP Delta Score")
    ax.set_title("CLIP Similarity Deltas per Direction", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Label summary table saved to {save_path}")

    return fig


def create_variance_plot(
    explained_variance_ratio: torch.Tensor,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot explained variance ratio for PCA components.

    Args:
        explained_variance_ratio: (K, T) tensor
        save_path: Path to save PNG
    """
    # Average across timesteps
    avg_var = explained_variance_ratio.mean(dim=1).numpy()  # (K,)
    cumsum = np.cumsum(avg_var)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart of individual variance
    x = np.arange(len(avg_var))
    ax1.bar(x, avg_var, color="steelblue")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio (avg over timesteps)")
    ax1.set_title("Individual Explained Variance")
    ax1.set_xticks(x)

    # Cumulative variance
    ax2.plot(x, cumsum, "o-", color="steelblue")
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Explained Variance")
    ax2.set_title("Cumulative Explained Variance")
    ax2.set_xticks(x)
    ax2.axhline(y=0.9, color="red", linestyle="--", alpha=0.5, label="90%")
    ax2.legend()

    plt.tight_layout()

    if save_path:
        ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Variance plot saved to {save_path}")

    return fig
