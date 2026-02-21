"""Stage 3: Automated Labeling -- CLI entry point."""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, save_config
from src.labeling import CLIPLabeler, VLMLabeler
from src.utils import load_json, save_json, ensure_dir, free_gpu_memory, get_best_gpu


def main():
    parser = argparse.ArgumentParser(description="Stage 3: Automated Labeling")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--edit-metadata", type=str, default=None,
                        help="Path to edit_metadata.json from Stage 2")
    parser.add_argument("--skip-clip", action="store_true",
                        help="Skip CLIP labeling")
    parser.add_argument("--skip-vlm", action="store_true",
                        help="Skip VLM labeling")
    args = parser.parse_args()

    config = load_config(args.config)

    # Find edit metadata
    meta_path = args.edit_metadata
    if meta_path is None:
        meta_path = os.path.join(config.output_dir, "stage2", "edit_metadata.json")

    if not os.path.exists(meta_path):
        print(f"Error: Edit metadata not found at {meta_path}")
        print("Run Stage 2 first: python scripts/run_stage2.py")
        sys.exit(1)

    edit_metadata = load_json(meta_path)

    # Save config
    stage3_dir = os.path.join(config.output_dir, "stage3")
    ensure_dir(stage3_dir)
    save_config(config, os.path.join(stage3_dir, "config.yaml"))

    device = config.model.device
    if device == "auto":
        device = get_best_gpu()

    clip_results = None
    vlm_results = None

    # Stage 3A: CLIP labeling
    if not args.skip_clip:
        print("\n=== Stage 3A: CLIP Labeling ===")
        clip_labeler = CLIPLabeler(config.labeling, device)
        clip_labeler.load_model()
        clip_results = clip_labeler.label_directions(edit_metadata, config.output_dir)
        clip_labeler.unload_model()

        save_json(clip_results, os.path.join(stage3_dir, "clip_scores.json"))
        print("CLIP results saved")

        # Print summary
        print("\nCLIP Top Labels:")
        for k, v in sorted(clip_results.items()):
            print(f"  {k}: {v['top_label']} (score: {v['top_score']:.4f})")

    # Stage 3B: VLM labeling
    if not args.skip_vlm:
        print("\n=== Stage 3B: VLM Labeling ===")
        vlm_labeler = VLMLabeler(config.labeling, device)
        vlm_labeler.load_model()
        vlm_results = vlm_labeler.caption_directions(edit_metadata, config.output_dir)
        vlm_labeler.unload_model()

        save_json(vlm_results, os.path.join(stage3_dir, "vlm_captions.json"))
        print("VLM results saved")

        # Print summary
        print("\nVLM Consensus Labels:")
        for k, v in sorted(vlm_results.items()):
            print(f"  {k}: {v['consensus_label']}")

    # Combine into summary
    summary = {}
    directions = set()
    if clip_results:
        directions.update(clip_results.keys())
    if vlm_results:
        directions.update(vlm_results.keys())

    for d in sorted(directions):
        entry = {"direction": d}
        if clip_results and d in clip_results:
            entry["clip_label"] = clip_results[d]["top_label"]
            entry["clip_score"] = clip_results[d]["top_score"]
        if vlm_results and d in vlm_results:
            entry["vlm_label"] = vlm_results[d]["consensus_label"]
            entry["vlm_captions"] = vlm_results[d]["captions"]
        summary[d] = entry

    save_json(summary, os.path.join(stage3_dir, "labels_summary.json"))
    print(f"\nStage 3 complete. Summary saved to {stage3_dir}/labels_summary.json")


if __name__ == "__main__":
    main()
