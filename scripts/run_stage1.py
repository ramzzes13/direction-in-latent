"""Stage 1: Direction Discovery -- CLI entry point."""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, save_config
from src.model import load_ddpm_model
from src.discovery import DirectionDiscovery
from src.visualization import create_variance_plot
from src.utils import ensure_dir


def main():
    parser = argparse.ArgumentParser(description="Stage 1: PCA Direction Discovery")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Override number of samples for PCA")
    parser.add_argument("--num-steps", type=int, default=None,
                        help="Override number of DDIM steps")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.num_samples is not None:
        config.discovery.num_samples = args.num_samples
    if args.num_steps is not None:
        config.sampler.num_inference_steps = args.num_steps

    # Save config for reproducibility
    ensure_dir(os.path.join(config.output_dir, "stage1"))
    save_config(config, os.path.join(config.output_dir, "stage1", "config.yaml"))

    # Load model
    unet, scheduler = load_ddpm_model(config.model)

    # Run discovery
    discovery = DirectionDiscovery(unet, scheduler, config)
    result = discovery.run()

    # Generate variance plot
    create_variance_plot(
        result["explained_variance_ratio"],
        save_path=os.path.join(config.output_dir, "stage1", "variance_plot.png"),
    )

    print("Stage 1 complete.")


if __name__ == "__main__":
    main()
