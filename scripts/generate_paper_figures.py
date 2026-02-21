"""Generate all publication-quality figures for the ICML paper.

All figures use real experimental data from the outputs/ directory.
"""

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
import matplotlib.gridspec as gridspec

from src.utils import load_json, load_checkpoint, ensure_dir

# Use consistent style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

OUTPUT_DIR = "outputs"
FIG_DIR = "paper/figures"


def fig1_pipeline_overview():
    """Figure 1: Pipeline overview - edit grid showing top 5 directions.

    Shows negative / original / positive for the top 5 PCA directions.
    """
    edit_meta = load_json(os.path.join(OUTPUT_DIR, "stage2", "edit_metadata.json"))
    clip_scores = load_json(os.path.join(OUTPUT_DIR, "stage3", "clip_scores.json"))

    num_dirs = 5
    seed_idx = 0  # Use first seed for cleaner visualization

    fig, axes = plt.subplots(num_dirs, 3, figsize=(6.5, 10))

    col_titles = [r"Negative ($-\alpha$)", "Original", r"Positive ($+\alpha$)"]

    for k in range(num_dirs):
        # Get edits for this direction
        dir_edits = [e for e in edit_meta["edits"] if e["direction_idx"] == k]
        e = dir_edits[seed_idx]

        neg = Image.open(os.path.join(OUTPUT_DIR, e["neg_path"]))
        orig = Image.open(os.path.join(OUTPUT_DIR, e["orig_path"]))
        pos = Image.open(os.path.join(OUTPUT_DIR, e["pos_path"]))

        for col, img in enumerate([neg, orig, pos]):
            axes[k, col].imshow(np.array(img))
            axes[k, col].axis("off")

        # Add direction label
        label = clip_scores[f"direction_{k}"]["top_label"]
        score = abs(clip_scores[f"direction_{k}"]["top_score"])
        short_label = label.replace("a person with ", "").replace("a face ", "").replace("a ", "").replace(" person", "")
        axes[k, 0].set_ylabel(f"PC{k}\n({short_label})", fontsize=9, rotation=0,
                               labelpad=60, va="center")

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10)

    fig.suptitle(r"Semantic Edits via $h$-space PCA Directions ($\alpha=5.0$, DDIM $\eta=0$)",
                 fontsize=11, y=0.98)
    plt.tight_layout(rect=[0.12, 0, 1, 0.96])
    fig.savefig(os.path.join(FIG_DIR, "fig1_edit_grid.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "fig1_edit_grid.png"))
    plt.close(fig)
    print("Figure 1: Edit grid saved")


def fig2_variance_plot():
    """Figure 2: PCA explained variance ratio."""
    pca_data = load_checkpoint(os.path.join(OUTPUT_DIR, "stage1", "pca_components.pt"))
    var_ratio = pca_data["explained_variance_ratio"]  # (K, T)
    avg_var = var_ratio.mean(dim=1).numpy()  # (K,)
    cumsum = np.cumsum(avg_var)
    K = len(avg_var)
    x = np.arange(K)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8))

    ax1.bar(x, avg_var * 100, color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance (%)")
    ax1.set_title("(a) Individual Components")
    ax1.set_xticks(x[::2])
    ax1.set_xticklabels([str(i) for i in x[::2]])

    ax2.plot(x, cumsum * 100, "o-", color="#4C72B0", markersize=3, linewidth=1.5)
    ax2.axhline(y=90, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
    ax2.text(K - 1, 91, "90%", fontsize=7, color="red", ha="right")
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Variance (%)")
    ax2.set_title("(b) Cumulative Variance")
    ax2.set_xticks(x[::2])
    ax2.set_xticklabels([str(i) for i in x[::2]])

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_pca_variance.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "fig2_pca_variance.png"))
    plt.close(fig)
    print("Figure 2: PCA variance saved")


def fig3_clip_heatmap():
    """Figure 3: CLIP delta score heatmap (directions x attributes)."""
    clip_scores = load_json(os.path.join(OUTPUT_DIR, "stage3", "clip_scores.json"))

    directions = sorted(clip_scores.keys())
    attributes = list(clip_scores[directions[0]]["scores"].keys())

    # Build matrix
    matrix = np.zeros((len(directions), len(attributes)))
    for i, d in enumerate(directions):
        for j, attr in enumerate(attributes):
            matrix[i, j] = clip_scores[d]["scores"].get(attr, 0.0)

    # Short labels
    short_attrs = []
    for a in attributes:
        a = a.replace("a person ", "").replace("a face ", "").replace("a ", "")
        a = a.replace("with ", "").replace("without ", "no ")
        short_attrs.append(a)

    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto",
                   vmin=-0.9, vmax=0.9, interpolation="nearest")

    ax.set_xticks(range(len(attributes)))
    ax.set_xticklabels(short_attrs, rotation=45, ha="right", fontsize=6.5)
    ax.set_yticks(range(len(directions)))
    ax.set_yticklabels([f"PC{d.replace('direction_', '')}" for d in directions], fontsize=8)

    # Highlight top label per direction
    for i, d in enumerate(directions):
        top_attr = clip_scores[d]["top_label"]
        j = attributes.index(top_attr)
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                   fill=False, edgecolor="black", linewidth=1.5))

    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(r"$\Delta S$ (logit-scaled)", fontsize=8)
    ax.set_title("CLIP Similarity Delta per Direction and Attribute", fontsize=10)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_clip_heatmap.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "fig3_clip_heatmap.png"))
    plt.close(fig)
    print("Figure 3: CLIP heatmap saved")


def fig4_alpha_sensitivity():
    """Figure 4: Alpha sensitivity - CLIP confidence vs edit strength."""
    results = load_json(os.path.join(OUTPUT_DIR, "experiment4_alpha", "alpha_sensitivity_results.json"))

    alpha_values = sorted([float(k) for k in results.keys()])
    num_dirs = 5

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8))

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

    # Per-direction curves
    for k in range(num_dirs):
        scores = []
        for alpha in alpha_values:
            key = f"direction_{k}"
            entry = results[str(alpha)]["clip_scores"].get(key, {})
            scores.append(abs(entry.get("top_score", 0)))
        ax1.plot(alpha_values, scores, "o-", label=f"PC{k}",
                color=colors[k], markersize=4, linewidth=1.5)

    ax1.set_xlabel(r"Edit strength $\alpha$")
    ax1.set_ylabel(r"$|\Delta S|$ (CLIP confidence)")
    ax1.set_title("(a) Per-Direction Confidence")
    ax1.legend(fontsize=7, ncol=2, loc="upper left")
    ax1.grid(True, alpha=0.2)

    # Mean across directions
    mean_scores = []
    std_scores = []
    for alpha in alpha_values:
        dir_scores = []
        for k in range(num_dirs):
            key = f"direction_{k}"
            s = results[str(alpha)]["clip_scores"].get(key, {}).get("top_score", 0)
            dir_scores.append(abs(s))
        mean_scores.append(np.mean(dir_scores))
        std_scores.append(np.std(dir_scores))

    ax2.errorbar(alpha_values, mean_scores, yerr=std_scores,
                fmt="s-", color="#4C72B0", markersize=5, linewidth=1.5,
                capsize=3, capthick=1)
    ax2.set_xlabel(r"Edit strength $\alpha$")
    ax2.set_ylabel(r"Mean $|\Delta S|$")
    ax2.set_title(r"(b) Average Confidence ($\pm$ std)")
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_alpha_sensitivity.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "fig4_alpha_sensitivity.png"))
    plt.close(fig)
    print("Figure 4: Alpha sensitivity saved")


def fig5_eta_comparison():
    """Figure 5: Comparison of eta=0 (DDIM) vs eta=1 (DDPM)."""
    comparison = load_json(os.path.join(OUTPUT_DIR, "comparison_eta", "eta_comparison.json"))

    eta0 = comparison["eta0"]
    eta1 = comparison["eta1"]
    directions = sorted(eta0.keys())

    scores0 = [abs(eta0[d]["top_score"]) for d in directions]
    scores1 = [abs(eta1[d]["top_score"]) for d in directions]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8))

    x = np.arange(len(directions))
    width = 0.35

    ax1.bar(x - width/2, scores0, width, label=r"$\eta=0$ (DDIM)", color="#4C72B0")
    ax1.bar(x + width/2, scores1, width, label=r"$\eta=1$ (DDPM)", color="#DD8452")
    ax1.set_xlabel("Direction")
    ax1.set_ylabel(r"$|\Delta S|$")
    ax1.set_title("(a) Score Magnitude")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"PC{d.replace('direction_', '')}" for d in directions], fontsize=7)
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.2, axis="y")

    # Label diversity comparison
    labels0 = [eta0[d]["top_label"] for d in directions]
    labels1 = [eta1[d]["top_label"] for d in directions]
    counts0 = Counter(labels0)
    counts1 = Counter(labels1)

    all_labels = sorted(set(labels0 + labels1))
    short_labels = []
    for l in all_labels:
        l = l.replace("a person ", "").replace("a face ", "").replace("a ", "")
        l = l.replace("with ", "").replace("without ", "no ")
        short_labels.append(l[:15])

    y_pos = np.arange(len(all_labels))
    h = 0.35

    vals0 = [counts0.get(l, 0) for l in all_labels]
    vals1 = [counts1.get(l, 0) for l in all_labels]

    ax2.barh(y_pos - h/2, vals0, h, label=r"$\eta=0$ (DDIM)", color="#4C72B0")
    ax2.barh(y_pos + h/2, vals1, h, label=r"$\eta=1$ (DDPM)", color="#DD8452")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(short_labels, fontsize=6.5)
    ax2.set_xlabel("Count")
    ax2.set_title("(b) Label Distribution")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.2, axis="x")

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig5_eta_comparison.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "fig5_eta_comparison.png"))
    plt.close(fig)
    print("Figure 5: Eta comparison saved")


def fig6_edit_magnitude():
    """Figure 6: Edit magnitude per direction (L2 pixel difference)."""
    report = load_json(os.path.join(OUTPUT_DIR, "analysis", "analysis_report.json"))
    eq = report["edit_quality"]

    dirs = sorted(eq.keys(), key=int)
    pos_vals = [eq[k]["mean_l2_pos"] for k in dirs]
    neg_vals = [eq[k]["mean_l2_neg"] for k in dirs]

    fig, ax = plt.subplots(figsize=(3.25, 2.5))

    x = np.arange(len(dirs))
    width = 0.35

    ax.bar(x - width/2, pos_vals, width, label=r"+$\alpha$", color="#4C72B0")
    ax.bar(x + width/2, neg_vals, width, label=r"-$\alpha$", color="#DD8452")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Mean L2 Pixel Difference")
    ax.set_title("Edit Magnitude per Direction")
    ax.set_xticks(x)
    ax.set_xticklabels([f"PC{k}" for k in dirs], fontsize=7)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig6_edit_magnitude.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "fig6_edit_magnitude.png"))
    plt.close(fig)
    print("Figure 6: Edit magnitude saved")


def fig7_per_seed_consistency():
    """Figure 7: Per-seed consistency of CLIP labeling."""
    clip_scores = load_json(os.path.join(OUTPUT_DIR, "stage3", "clip_scores.json"))

    directions = sorted(clip_scores.keys())

    consistency = []
    for d in directions:
        entry = clip_scores[d]
        top_label = entry["top_label"]
        seed_deltas = entry.get("per_seed_deltas", [])
        if not seed_deltas:
            consistency.append(0)
            continue
        consistent = 0
        for sd in seed_deltas:
            seed_top = max(sd.keys(), key=lambda k: abs(sd[k]))
            if seed_top == top_label:
                consistent += 1
        consistency.append(consistent / len(seed_deltas) * 100)

    fig, ax = plt.subplots(figsize=(3.25, 2.5))

    x = np.arange(len(directions))
    colors = ["#55A868" if c >= 60 else "#DD8452" if c >= 40 else "#C44E52" for c in consistency]

    ax.bar(x, consistency, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(y=60, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.set_xlabel("Direction")
    ax.set_ylabel("Seed Consistency (%)")
    ax.set_title("CLIP Label Agreement Across Seeds")
    ax.set_xticks(x)
    ax.set_xticklabels([f"PC{d.replace('direction_', '')}" for d in directions], fontsize=7)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig7_seed_consistency.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "fig7_seed_consistency.png"))
    plt.close(fig)
    print("Figure 7: Seed consistency saved")


def fig8_variance_by_timestep():
    """Figure 8: How PCA variance changes across timesteps for top components."""
    pca_data = load_checkpoint(os.path.join(OUTPUT_DIR, "stage1", "pca_components.pt"))
    var_ratio = pca_data["explained_variance_ratio"]  # (K, T)

    fig, ax = plt.subplots(figsize=(3.25, 2.5))

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
    T = var_ratio.shape[1]
    timesteps = np.arange(T)

    for k in range(5):
        ax.plot(timesteps, var_ratio[k].numpy() * 100,
                color=colors[k], linewidth=1.2, label=f"PC{k}", alpha=0.8)

    ax.set_xlabel("Timestep index")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_title("Variance by Timestep")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig8_variance_timestep.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "fig8_variance_timestep.png"))
    plt.close(fig)
    print("Figure 8: Variance by timestep saved")


def table_data():
    """Print data for tables to be included in the paper."""
    clip_scores = load_json(os.path.join(OUTPUT_DIR, "stage3", "clip_scores.json"))
    vlm_captions = load_json(os.path.join(OUTPUT_DIR, "stage3", "vlm_captions.json"))
    report = load_json(os.path.join(OUTPUT_DIR, "analysis", "analysis_report.json"))
    alpha_results = load_json(os.path.join(OUTPUT_DIR, "experiment4_alpha", "alpha_sensitivity_results.json"))
    eta_comparison = load_json(os.path.join(OUTPUT_DIR, "comparison_eta", "eta_comparison.json"))

    print("\n=== TABLE 1: CLIP Labeling Results (Top-10 Directions) ===")
    print("Dir | Top Label | Score | Specific Label | Score | Consistency")
    for d in sorted(clip_scores.keys()):
        entry = clip_scores[d]
        top = entry["top_label"]
        ts = entry["top_score"]
        spec = entry.get("specific_label", top)
        ss = entry.get("specific_score", ts)

        # Consistency
        seed_deltas = entry.get("per_seed_deltas", [])
        cons = "N/A"
        if seed_deltas:
            agree = sum(1 for sd in seed_deltas
                       if max(sd.keys(), key=lambda k: abs(sd[k])) == top)
            cons = f"{agree}/{len(seed_deltas)}"

        pc = d.replace("direction_", "")
        print(f"PC{pc} | {top} | {ts:+.3f} | {spec} | {ss:+.3f} | {cons}")

    print("\n=== TABLE 2: Alpha Sensitivity ===")
    alpha_values = sorted([float(k) for k in alpha_results.keys()])
    for alpha in alpha_values:
        key = str(alpha)
        cs = alpha_results[key]["clip_scores"]
        mean_abs = np.mean([abs(cs[f"direction_{k}"]["top_score"]) for k in range(5)])
        print(f"alpha={alpha:.1f}: mean_abs={mean_abs:.3f}")

    print("\n=== TABLE 3: Eta Comparison Summary ===")
    print(f"eta=0: unique={eta_comparison['summary']['eta0_unique_labels']}, "
          f"mean_score={eta_comparison['summary']['eta0_mean_score']:.3f}")
    print(f"eta=1: unique={eta_comparison['summary']['eta1_unique_labels']}, "
          f"mean_score={eta_comparison['summary']['eta1_mean_score']:.3f}")

    print("\n=== PCA Variance Summary ===")
    pca = report["pca"]
    for k, v in enumerate(pca["avg_variance_ratio"][:10]):
        print(f"PC{k}: {v:.4f} (cumulative: {pca['cumulative_variance'][k]:.4f})")


if __name__ == "__main__":
    ensure_dir(FIG_DIR)

    print("Generating publication-quality figures...")
    print("=" * 60)

    fig1_pipeline_overview()
    fig2_variance_plot()
    fig3_clip_heatmap()
    fig4_alpha_sensitivity()
    fig5_eta_comparison()
    fig6_edit_magnitude()
    fig7_per_seed_consistency()
    fig8_variance_by_timestep()
    table_data()

    print("\n" + "=" * 60)
    print(f"All figures saved to {FIG_DIR}/")
