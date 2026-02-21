"""Stage 2: Edit Generation -- CLI entry point."""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, save_config
from src.model import load_ddpm_model
from src.editing import EditGenerator
from src.visualization import create_edit_grid
from src.utils import load_checkpoint, ensure_dir, tensor_to_pil

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Edit Generation")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--pca-checkpoint", type=str, default=None,
                        help="Path to pca_components.pt from Stage 1")
    args = parser.parse_args()

    config = load_config(args.config)

    # Find PCA checkpoint
    pca_path = args.pca_checkpoint
    if pca_path is None:
        pca_path = os.path.join(config.output_dir, "stage1", "pca_components.pt")

    if not os.path.exists(pca_path):
        print(f"Error: PCA checkpoint not found at {pca_path}")
        print("Run Stage 1 first: python scripts/run_stage1.py")
        sys.exit(1)

    pca_data = load_checkpoint(pca_path)

    # Save config
    ensure_dir(os.path.join(config.output_dir, "stage2"))
    save_config(config, os.path.join(config.output_dir, "stage2", "config.yaml"))

    # Load model
    unet, scheduler = load_ddpm_model(config.model)

    # Generate edits
    editor = EditGenerator(unet, scheduler, config)
    metadata = editor.generate_edit_pairs(pca_data)

    # Create grid visualizations
    grids_dir = os.path.join(config.output_dir, "stage2", "grids")
    ensure_dir(grids_dir)

    for k in range(config.edit.num_directions):
        direction_edits = [e for e in metadata["edits"] if e["direction_idx"] == k]
        images = []
        for e in direction_edits:
            neg = Image.open(os.path.join(config.output_dir, e["neg_path"]))
            orig = Image.open(os.path.join(config.output_dir, e["orig_path"]))
            pos = Image.open(os.path.join(config.output_dir, e["pos_path"]))
            images.append((neg, orig, pos))

        create_edit_grid(
            direction_idx=k,
            images=images,
            alpha=metadata["alpha"],
            save_path=os.path.join(grids_dir, f"pc{k:02d}_grid.png"),
        )

    print("Stage 2 complete.")


if __name__ == "__main__":
    main()
