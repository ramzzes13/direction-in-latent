"""Comprehensive analysis and evaluation of the pipeline results.

Generates:
1. CLIP score analysis with statistical significance
2. Direction diversity metrics
3. Comparison of CLIP vs VLM labeling
4. Edit quality assessment (pixel-level change analysis)
5. Summary report
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from PIL import Image
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import load_config
from src.utils import load_json, load_checkpoint, save_json, ensure_dir


def analyze_clip_scores(clip_results, output_dir):
    """Analyze CLIP labeling results in detail."""
    print("\n" + "=" * 60)
    print("CLIP LABELING ANALYSIS")
    print("=" * 60)

    directions = sorted(clip_results.keys())
    attributes = list(clip_results[directions[0]]["scores"].keys())

    # Print detailed scores table
    print(f"\n{'Direction':<15} {'Top Label':<30} {'Score':>8} {'|Score|':>8}")
    print("-" * 65)

    labels_assigned = []
    abs_scores = []

    for d in directions:
        entry = clip_results[d]
        label = entry["top_label"]
        score = entry["top_score"]
        abs_score = entry.get("top_abs_score", abs(score))
        labels_assigned.append(label)
        abs_scores.append(abs_score)

        short_label = label[:28]
        print(f"{d:<15} {short_label:<30} {score:>+8.4f} {abs_score:>8.4f}")

    # Label diversity
    unique_labels = set(labels_assigned)
    label_counts = Counter(labels_assigned)

    print(f"\n--- Label Diversity ---")
    print(f"Total directions: {len(directions)}")
    print(f"Unique labels assigned: {len(unique_labels)}")
    print(f"Label diversity ratio: {len(unique_labels)/len(directions):.2%}")

    if len(label_counts) > 0:
        print(f"\nLabel frequency:")
        for label, count in label_counts.most_common():
            short = label[:35]
            print(f"  {short:<38} {count}x")

    # Score statistics
    all_mean_scores = []
    for d in directions:
        scores = list(clip_results[d]["scores"].values())
        all_mean_scores.extend(scores)

    print(f"\n--- Score Statistics ---")
    print(f"Mean absolute top score: {np.mean(abs_scores):.4f}")
    print(f"Std absolute top score: {np.std(abs_scores):.4f}")
    print(f"Max absolute top score: {np.max(abs_scores):.4f}")
    print(f"Min absolute top score: {np.min(abs_scores):.4f}")

    # Per-seed consistency
    print(f"\n--- Per-Seed Consistency ---")
    for d in directions:
        seed_deltas = clip_results[d].get("per_seed_deltas", [])
        if not seed_deltas:
            continue
        # Check if top label is consistent across seeds
        top_label = clip_results[d]["top_label"]
        consistent = 0
        for sd in seed_deltas:
            seed_top = max(sd.keys(), key=lambda k: abs(sd[k]))
            if seed_top == top_label:
                consistent += 1
        pct = consistent / len(seed_deltas) * 100
        print(f"  {d}: {consistent}/{len(seed_deltas)} seeds agree ({pct:.0f}%)")

    return {
        "num_directions": len(directions),
        "unique_labels": len(unique_labels),
        "diversity_ratio": len(unique_labels) / len(directions),
        "mean_abs_score": float(np.mean(abs_scores)),
        "std_abs_score": float(np.std(abs_scores)),
        "label_counts": dict(label_counts),
    }


def analyze_vlm_results(vlm_results, output_dir):
    """Analyze VLM captioning results."""
    if vlm_results is None:
        print("\n[VLM results not available]")
        return None

    print("\n" + "=" * 60)
    print("VLM LABELING ANALYSIS")
    print("=" * 60)

    for d in sorted(vlm_results.keys()):
        entry = vlm_results[d]
        consensus = entry.get("consensus_label", "N/A")
        captions = entry.get("captions", [])
        print(f"\n{d}: {consensus}")
        for i, cap in enumerate(captions):
            print(f"  Seed {i}: {cap[:100]}")

    return {"num_directions": len(vlm_results)}


def compare_clip_vlm(clip_results, vlm_results, output_dir):
    """Compare CLIP and VLM labeling approaches."""
    if vlm_results is None:
        return None

    print("\n" + "=" * 60)
    print("CLIP vs VLM COMPARISON")
    print("=" * 60)

    print(f"\n{'Direction':<15} {'CLIP Label':<30} {'VLM Label':<30}")
    print("-" * 78)

    for d in sorted(clip_results.keys()):
        clip_label = clip_results[d]["top_label"][:28]
        vlm_label = vlm_results.get(d, {}).get("consensus_label", "N/A")[:28]
        print(f"{d:<15} {clip_label:<30} {vlm_label:<30}")


def analyze_edit_quality(edit_metadata, config_output_dir, analysis_dir):
    """Analyze the magnitude of edits by computing pixel differences."""
    print("\n" + "=" * 60)
    print("EDIT QUALITY ANALYSIS")
    print("=" * 60)

    edits_by_dir = {}
    for e in edit_metadata["edits"]:
        k = e["direction_idx"]
        if k not in edits_by_dir:
            edits_by_dir[k] = []
        edits_by_dir[k].append(e)

    dir_diffs = {}

    for k in sorted(edits_by_dir.keys()):
        l2_diffs_pos = []
        l2_diffs_neg = []

        for e in edits_by_dir[k]:
            orig = np.array(Image.open(os.path.join(config_output_dir, e["orig_path"]))).astype(float)
            pos = np.array(Image.open(os.path.join(config_output_dir, e["pos_path"]))).astype(float)
            neg = np.array(Image.open(os.path.join(config_output_dir, e["neg_path"]))).astype(float)

            l2_pos = np.sqrt(np.mean((pos - orig) ** 2))
            l2_neg = np.sqrt(np.mean((neg - orig) ** 2))
            l2_diffs_pos.append(l2_pos)
            l2_diffs_neg.append(l2_neg)

        mean_pos = np.mean(l2_diffs_pos)
        mean_neg = np.mean(l2_diffs_neg)
        dir_diffs[k] = {
            "mean_l2_pos": float(mean_pos),
            "mean_l2_neg": float(mean_neg),
            "mean_l2_avg": float((mean_pos + mean_neg) / 2),
        }

        print(f"  PC{k}: L2(orig,pos)={mean_pos:.2f}, L2(orig,neg)={mean_neg:.2f}")

    # Plot edit magnitudes
    fig, ax = plt.subplots(figsize=(10, 5))
    dirs = sorted(dir_diffs.keys())
    pos_vals = [dir_diffs[k]["mean_l2_pos"] for k in dirs]
    neg_vals = [dir_diffs[k]["mean_l2_neg"] for k in dirs]

    x = np.arange(len(dirs))
    width = 0.35
    ax.bar(x - width/2, pos_vals, width, label="Positive edit", color="steelblue")
    ax.bar(x + width/2, neg_vals, width, label="Negative edit", color="coral")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Mean L2 Pixel Difference")
    ax.set_title("Edit Magnitude per Direction")
    ax.set_xticks(x)
    ax.set_xticklabels([f"PC{k}" for k in dirs])
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(analysis_dir, "edit_magnitudes.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    return dir_diffs


def analyze_pca_variance(pca_data, analysis_dir):
    """Analyze PCA explained variance in detail."""
    print("\n" + "=" * 60)
    print("PCA VARIANCE ANALYSIS")
    print("=" * 60)

    var_ratio = pca_data["explained_variance_ratio"]  # (K, T)
    avg_var = var_ratio.mean(dim=1).numpy()
    cumsum = np.cumsum(avg_var)

    print(f"Number of components: {var_ratio.shape[0]}")
    print(f"Number of timesteps: {var_ratio.shape[1]}")
    print(f"Number of samples used: {pca_data.get('num_samples', 'unknown')}")

    print(f"\nPer-component explained variance (averaged over timesteps):")
    for k in range(len(avg_var)):
        print(f"  PC{k}: {avg_var[k]:.4f} (cumulative: {cumsum[k]:.4f})")

    # Find 90% variance threshold
    threshold_90 = np.searchsorted(cumsum, 0.9) + 1
    print(f"\nComponents needed for 90% variance: {threshold_90}")

    # Per-timestep analysis
    print(f"\nPC0 variance by timestep range:")
    T = var_ratio.shape[1]
    for start, end, label in [(0, T//4, "early"), (T//4, T//2, "mid-early"),
                                (T//2, 3*T//4, "mid-late"), (3*T//4, T, "late")]:
        mean_var = var_ratio[0, start:end].mean().item()
        print(f"  {label} (t={start}-{end}): {mean_var:.4f}")

    return {
        "avg_variance_ratio": avg_var.tolist(),
        "cumulative_variance": cumsum.tolist(),
        "components_for_90pct": int(threshold_90),
    }


def create_summary_report(clip_analysis, vlm_analysis, edit_quality,
                          pca_analysis, clip_results, vlm_results, analysis_dir):
    """Create a summary report."""
    print("\n" + "=" * 60)
    print("SUMMARY REPORT")
    print("=" * 60)

    report = {
        "pca": pca_analysis,
        "clip_analysis": clip_analysis,
        "edit_quality": edit_quality,
        "direction_labels": {},
    }

    for d in sorted(clip_results.keys()):
        entry = {
            "clip_label": clip_results[d]["top_label"],
            "clip_score": clip_results[d]["top_score"],
            "clip_abs_score": clip_results[d].get("top_abs_score", abs(clip_results[d]["top_score"])),
        }
        if "top_positive" in clip_results[d]:
            entry["clip_positive"] = clip_results[d]["top_positive"]
            entry["clip_negative"] = clip_results[d]["top_negative"]
        if vlm_results and d in vlm_results:
            entry["vlm_label"] = vlm_results[d].get("consensus_label", "N/A")
        report["direction_labels"][d] = entry

    save_json(report, os.path.join(analysis_dir, "analysis_report.json"))

    # Print final direction assignments
    print(f"\n{'PC':<6} {'CLIP Label':<30} {'Score':>8} {'VLM Label':<30}")
    print("-" * 78)
    for d in sorted(report["direction_labels"].keys()):
        entry = report["direction_labels"][d]
        pc = d.replace("direction_", "")
        clip_l = entry["clip_label"][:28]
        score = entry["clip_score"]
        vlm_l = entry.get("vlm_label", "N/A")[:28]
        print(f"PC{pc:<4} {clip_l:<30} {score:>+8.4f} {vlm_l:<30}")

    print(f"\nAnalysis complete. Report saved to {analysis_dir}/analysis_report.json")
    return report


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Analysis")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    analysis_dir = os.path.join(config.output_dir, "analysis")
    ensure_dir(analysis_dir)

    # Load all results
    clip_path = os.path.join(config.output_dir, "stage3", "clip_scores.json")
    vlm_path = os.path.join(config.output_dir, "stage3", "vlm_captions.json")
    meta_path = os.path.join(config.output_dir, "stage2", "edit_metadata.json")
    pca_path = os.path.join(config.output_dir, "stage1", "pca_components.pt")

    clip_results = load_json(clip_path) if os.path.exists(clip_path) else None
    vlm_results = load_json(vlm_path) if os.path.exists(vlm_path) else None
    edit_metadata = load_json(meta_path) if os.path.exists(meta_path) else None
    pca_data = load_checkpoint(pca_path) if os.path.exists(pca_path) else None

    # Run analyses
    pca_analysis = None
    if pca_data:
        pca_analysis = analyze_pca_variance(pca_data, analysis_dir)

    clip_analysis = None
    if clip_results:
        clip_analysis = analyze_clip_scores(clip_results, analysis_dir)

    vlm_analysis = None
    if vlm_results:
        vlm_analysis = analyze_vlm_results(vlm_results, analysis_dir)

    if clip_results and vlm_results:
        compare_clip_vlm(clip_results, vlm_results, analysis_dir)

    edit_quality = None
    if edit_metadata:
        edit_quality = analyze_edit_quality(
            edit_metadata, config.output_dir, analysis_dir
        )

    # Create summary
    if clip_results:
        create_summary_report(
            clip_analysis, vlm_analysis, edit_quality,
            pca_analysis, clip_results, vlm_results, analysis_dir,
        )


if __name__ == "__main__":
    main()
