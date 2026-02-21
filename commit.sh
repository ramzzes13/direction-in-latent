#!/bin/bash
set -e
cd /home/mekashirskiy/rom4ik/direction-in-latent

git add .gitignore requirements.txt configs/default.yaml \
  src/__init__.py src/config.py src/utils.py src/model.py src/hooks.py \
  src/sampler.py src/discovery.py src/editing.py src/labeling.py src/visualization.py \
  scripts/run_pipeline.py scripts/run_stage1.py scripts/run_stage2.py scripts/run_stage3.py

git commit -m "Implement full pipeline for automated interpretation of semantic latent directions

Three-stage pipeline:
- Stage 1: PCA direction discovery on DDPM bottleneck activations (500 samples)
- Stage 2: Edit generation with positive/negative h-space perturbations (10 dirs x 5 seeds)
- Stage 3: Automated labeling via CLIP zero-shot + BLIP-2 VLM captioning

Full-scale run completed:
- PC0: face rotation (CLIP confirmed)
- PC4: eyes closed
- PC6: bald
- PC8: glasses

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"

git push origin main

echo "Commit and push complete!"
