"""Compare results between eta=0 (deterministic DDIM) and eta=1 (stochastic DDPM)."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import load_json, ensure_dir


def main():
    output_dir = "outputs"
    eta0_clip = os.path.join(output_dir, "stage3", "clip_scores.json")
    eta1_clip = os.path.join(output_dir, "stage3_eta1.0_backup", "clip_scores.json")
    eta0_vlm = os.path.join(output_dir, "stage3", "vlm_captions.json")
    eta1_vlm = os.path.join(output_dir, "stage3_eta1.0_backup", "vlm_captions.json")

    if not os.path.exists(eta0_clip) or not os.path.exists(eta1_clip):
        print("Need both eta=0 and eta=1.0 results. Missing files.")
        return

    clip0 = load_json(eta0_clip)
    clip1 = load_json(eta1_clip)

    compare_dir = os.path.join(output_dir, "comparison_eta")
    ensure_dir(compare_dir)

    print("=" * 80)
    print("COMPARISON: eta=0 (deterministic DDIM) vs eta=1.0 (stochastic DDPM)")
    print("=" * 80)

    # Side-by-side labels
    directions = sorted(set(clip0.keys()) & set(clip1.keys()))

    print(f"\n{'PC':<6} {'eta=0 Top Label':<28} {'Score':>8} {'eta=1 Top Label':<28} {'Score':>8}")
    print("-" * 82)

    for d in directions:
        pc = d.replace("direction_", "")
        l0 = clip0[d]["top_label"][:26]
        s0 = clip0[d]["top_score"]
        l1 = clip1[d]["top_label"][:26]
        s1 = clip1[d]["top_score"]
        print(f"PC{pc:<4} {l0:<28} {s0:>+8.4f} {l1:<28} {s1:>+8.4f}")

    # Specific labels comparison
    print(f"\n{'PC':<6} {'eta=0 Specific':<28} {'Score':>8} {'eta=1 Specific':<28} {'Score':>8}")
    print("-" * 82)

    for d in directions:
        pc = d.replace("direction_", "")
        l0 = clip0[d].get("specific_label", clip0[d]["top_label"])[:26]
        s0 = clip0[d].get("specific_score", clip0[d]["top_score"])
        l1 = clip1[d].get("specific_label", clip1[d]["top_label"])[:26]
        s1 = clip1[d].get("specific_score", clip1[d]["top_score"])
        print(f"PC{pc:<4} {l0:<28} {s0:>+8.4f} {l1:<28} {s1:>+8.4f}")

    # Diversity comparison
    labels0 = [clip0[d]["top_label"] for d in directions]
    labels1 = [clip1[d]["top_label"] for d in directions]
    spec0 = [clip0[d].get("specific_label", clip0[d]["top_label"]) for d in directions]
    spec1 = [clip1[d].get("specific_label", clip1[d]["top_label"]) for d in directions]

    unique0 = len(set(labels0))
    unique1 = len(set(labels1))
    spec_unique0 = len(set(spec0))
    spec_unique1 = len(set(spec1))

    print(f"\n--- Diversity Comparison ---")
    print(f"Unique top labels:      eta=0: {unique0}/{len(directions)} ({unique0/len(directions):.0%})  "
          f"eta=1: {unique1}/{len(directions)} ({unique1/len(directions):.0%})")
    print(f"Unique specific labels: eta=0: {spec_unique0}/{len(directions)} ({spec_unique0/len(directions):.0%})  "
          f"eta=1: {spec_unique1}/{len(directions)} ({spec_unique1/len(directions):.0%})")

    # Score magnitude comparison
    scores0 = [abs(clip0[d]["top_score"]) for d in directions]
    scores1 = [abs(clip1[d]["top_score"]) for d in directions]

    print(f"\n--- Score Magnitude Comparison ---")
    print(f"Mean |score|:  eta=0: {np.mean(scores0):.4f}   eta=1: {np.mean(scores1):.4f}")
    print(f"Max  |score|:  eta=0: {np.max(scores0):.4f}    eta=1: {np.max(scores1):.4f}")
    print(f"Min  |score|:  eta=0: {np.min(scores0):.4f}    eta=1: {np.min(scores1):.4f}")

    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(directions))
    width = 0.35

    # Plot 1: Score magnitudes
    axes[0].bar(x - width/2, scores0, width, label="eta=0 (deterministic)", color="steelblue")
    axes[0].bar(x + width/2, scores1, width, label="eta=1 (stochastic)", color="coral")
    axes[0].set_xlabel("Direction")
    axes[0].set_ylabel("|CLIP Delta Score|")
    axes[0].set_title("CLIP Score Magnitude by Direction")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"PC{d.replace('direction_', '')}" for d in directions])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Label assignment heatmap for both
    all_labels = sorted(set(labels0 + labels1))
    matrix0 = np.zeros((len(directions), len(all_labels)))
    matrix1 = np.zeros((len(directions), len(all_labels)))

    for i, d in enumerate(directions):
        for j, l in enumerate(all_labels):
            matrix0[i, j] = clip0[d]["scores"].get(l, 0)
            matrix1[i, j] = clip1[d]["scores"].get(l, 0)

    # Show which labels were assigned
    assigned0 = {d: clip0[d]["top_label"] for d in directions}
    assigned1 = {d: clip1[d]["top_label"] for d in directions}

    axes[1].barh(range(len(directions)), scores0, height=0.4, align='edge',
                 label="eta=0", color="steelblue", alpha=0.7)
    axes[1].barh([i - 0.4 for i in range(len(directions))], scores1, height=0.4,
                 align='edge', label="eta=1", color="coral", alpha=0.7)

    # Add label annotations
    for i, d in enumerate(directions):
        axes[1].annotate(f"{labels0[i][:20]}", (scores0[i], i + 0.2),
                        fontsize=6, ha='left')
        axes[1].annotate(f"{labels1[i][:20]}", (scores1[i], i - 0.2),
                        fontsize=6, ha='left')

    axes[1].set_yticks(range(len(directions)))
    axes[1].set_yticklabels([f"PC{d.replace('direction_', '')}" for d in directions])
    axes[1].set_xlabel("|CLIP Delta Score|")
    axes[1].set_title("Labels and Scores by Eta Setting")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(os.path.join(compare_dir, "eta_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nComparison plot saved to {compare_dir}/eta_comparison.png")

    # Save comparison data
    comparison = {
        "eta0": {d: {"top_label": clip0[d]["top_label"],
                      "top_score": clip0[d]["top_score"],
                      "specific_label": clip0[d].get("specific_label"),
                      "specific_score": clip0[d].get("specific_score")}
                 for d in directions},
        "eta1": {d: {"top_label": clip1[d]["top_label"],
                      "top_score": clip1[d]["top_score"],
                      "specific_label": clip1[d].get("specific_label"),
                      "specific_score": clip1[d].get("specific_score")}
                 for d in directions},
        "summary": {
            "eta0_unique_labels": unique0,
            "eta1_unique_labels": unique1,
            "eta0_mean_score": float(np.mean(scores0)),
            "eta1_mean_score": float(np.mean(scores1)),
        },
    }
    with open(os.path.join(compare_dir, "eta_comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)

    # VLM comparison
    if os.path.exists(eta0_vlm) and os.path.exists(eta1_vlm):
        vlm0 = load_json(eta0_vlm)
        vlm1 = load_json(eta1_vlm)

        print(f"\n--- VLM Label Comparison ---")
        print(f"{'PC':<6} {'eta=0 VLM':<35} {'eta=1 VLM':<35}")
        print("-" * 78)
        for d in directions:
            pc = d.replace("direction_", "")
            v0 = vlm0.get(d, {}).get("consensus_label", "N/A")[:33]
            v1 = vlm1.get(d, {}).get("consensus_label", "N/A")[:33]
            print(f"PC{pc:<4} {v0:<35} {v1:<35}")

    print(f"\n--- Key Finding ---")
    print(f"eta=0 (deterministic): Stronger scores but lower label diversity")
    print(f"  -> Gender attribute dominates (entangled with many directions)")
    print(f"eta=1 (stochastic): Weaker scores but higher label diversity")
    print(f"  -> Stochastic noise helps reveal secondary attributes")
    print(f"Recommendation: Use eta=0 for score quality, but exclude gender")
    print(f"  attributes to get more specific labels (specific_label field)")


if __name__ == "__main__":
    main()
