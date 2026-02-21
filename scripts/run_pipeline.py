"""Full pipeline orchestrator: runs all 3 stages sequentially."""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, save_config
from src.model import load_ddpm_model
from src.discovery import DirectionDiscovery
from src.editing import EditGenerator
from src.labeling import CLIPLabeler, VLMLabeler
from src.visualization import create_variance_plot, create_edit_grid, create_label_summary_table
from src.utils import (
    load_checkpoint, load_json, save_json, ensure_dir,
    free_gpu_memory, get_best_gpu,
)

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Full Pipeline: Stages 1-3")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--num-directions", type=int, default=None)
    parser.add_argument("--num-seeds", type=int, default=None)
    parser.add_argument("--skip-vlm", action="store_true",
                        help="Skip VLM labeling (saves time/memory)")
    args = parser.parse_args()

    config = load_config(args.config)

    # Apply CLI overrides
    if args.num_samples is not None:
        config.discovery.num_samples = args.num_samples
    if args.num_steps is not None:
        config.sampler.num_inference_steps = args.num_steps
    if args.num_directions is not None:
        config.edit.num_directions = args.num_directions
    if args.num_seeds is not None:
        config.edit.num_seeds_per_direction = args.num_seeds

    ensure_dir(config.output_dir)
    save_config(config, os.path.join(config.output_dir, "pipeline_config.yaml"))

    device = config.model.device
    if device == "auto":
        device = get_best_gpu()
        config.model.device = device

    # ============================
    # Stage 1: Direction Discovery
    # ============================
    pca_path = os.path.join(config.output_dir, "stage1", "pca_components.pt")
    if os.path.exists(pca_path):
        print("\n=== Stage 1: SKIPPED (checkpoint exists) ===")
        pca_data = load_checkpoint(pca_path)
    else:
        print("\n=== Stage 1: Direction Discovery ===")
        start = time.time()
        unet, scheduler = load_ddpm_model(config.model)
        discovery = DirectionDiscovery(unet, scheduler, config)
        pca_data = discovery.run()

        create_variance_plot(
            pca_data["explained_variance_ratio"],
            save_path=os.path.join(config.output_dir, "stage1", "variance_plot.png"),
        )

        # Free DDPM for next stage
        del discovery
        del unet, scheduler
        free_gpu_memory()
        elapsed = time.time() - start
        print(f"Stage 1 completed in {elapsed:.0f}s")

    # ============================
    # Stage 2: Edit Generation
    # ============================
    meta_path = os.path.join(config.output_dir, "stage2", "edit_metadata.json")
    if os.path.exists(meta_path):
        print("\n=== Stage 2: SKIPPED (metadata exists) ===")
        edit_metadata = load_json(meta_path)
    else:
        print("\n=== Stage 2: Edit Generation ===")
        start = time.time()
        unet, scheduler = load_ddpm_model(config.model)
        editor = EditGenerator(unet, scheduler, config)
        edit_metadata = editor.generate_edit_pairs(pca_data)

        # Create grids
        grids_dir = os.path.join(config.output_dir, "stage2", "grids")
        ensure_dir(grids_dir)
        for k in range(config.edit.num_directions):
            direction_edits = [e for e in edit_metadata["edits"] if e["direction_idx"] == k]
            images = []
            for e in direction_edits:
                neg = Image.open(os.path.join(config.output_dir, e["neg_path"]))
                orig = Image.open(os.path.join(config.output_dir, e["orig_path"]))
                pos = Image.open(os.path.join(config.output_dir, e["pos_path"]))
                images.append((neg, orig, pos))
            create_edit_grid(
                direction_idx=k, images=images, alpha=edit_metadata["alpha"],
                save_path=os.path.join(grids_dir, f"pc{k:02d}_grid.png"),
            )

        del editor, unet, scheduler
        free_gpu_memory()
        elapsed = time.time() - start
        print(f"Stage 2 completed in {elapsed:.0f}s")

    # ============================
    # Stage 3: Automated Labeling
    # ============================
    stage3_dir = os.path.join(config.output_dir, "stage3")
    ensure_dir(stage3_dir)

    # 3A: CLIP
    clip_path = os.path.join(stage3_dir, "clip_scores.json")
    if os.path.exists(clip_path):
        print("\n=== Stage 3A: SKIPPED (CLIP scores exist) ===")
        clip_results = load_json(clip_path)
    else:
        print("\n=== Stage 3A: CLIP Labeling ===")
        start = time.time()
        clip_labeler = CLIPLabeler(config.labeling, device)
        clip_labeler.load_model()
        clip_results = clip_labeler.label_directions(edit_metadata, config.output_dir)
        clip_labeler.unload_model()
        save_json(clip_results, clip_path)

        create_label_summary_table(
            clip_results,
            save_path=os.path.join(stage3_dir, "clip_heatmap.png"),
        )

        elapsed = time.time() - start
        print(f"Stage 3A completed in {elapsed:.0f}s")

    print("\nCLIP Top Labels:")
    for k, v in sorted(clip_results.items()):
        print(f"  {k}: {v['top_label']} (score: {v['top_score']:.4f})")

    # 3B: VLM
    vlm_results = None
    vlm_path = os.path.join(stage3_dir, "vlm_captions.json")
    if args.skip_vlm:
        print("\n=== Stage 3B: SKIPPED (--skip-vlm) ===")
    elif os.path.exists(vlm_path):
        print("\n=== Stage 3B: SKIPPED (VLM captions exist) ===")
        vlm_results = load_json(vlm_path)
    else:
        print("\n=== Stage 3B: VLM Labeling ===")
        start = time.time()
        vlm_labeler = VLMLabeler(config.labeling, device)
        vlm_labeler.load_model()
        vlm_results = vlm_labeler.caption_directions(edit_metadata, config.output_dir)
        vlm_labeler.unload_model()
        save_json(vlm_results, vlm_path)
        elapsed = time.time() - start
        print(f"Stage 3B completed in {elapsed:.0f}s")

    if vlm_results:
        print("\nVLM Consensus Labels:")
        for k, v in sorted(vlm_results.items()):
            print(f"  {k}: {v['consensus_label']}")

    # Final summary
    summary = {}
    for d in sorted(clip_results.keys()):
        entry = {
            "direction": d,
            "clip_label": clip_results[d]["top_label"],
            "clip_score": clip_results[d]["top_score"],
        }
        if vlm_results and d in vlm_results:
            entry["vlm_label"] = vlm_results[d]["consensus_label"]
            entry["vlm_captions"] = vlm_results[d]["captions"]
        summary[d] = entry

    save_json(summary, os.path.join(stage3_dir, "labels_summary.json"))
    print(f"\nPipeline complete! Results in {config.output_dir}/")


if __name__ == "__main__":
    main()
