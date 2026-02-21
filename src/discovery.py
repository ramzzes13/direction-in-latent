"""Stage 1: PCA direction discovery on h-space activations."""

import os
from typing import Dict

import numpy as np
import torch
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm

from src.config import PipelineConfig
from src.hooks import BottleneckHook
from src.sampler import generate_with_seed
from src.utils import save_checkpoint, ensure_dir


class DirectionDiscovery:
    """
    Discovers semantic directions via Incremental PCA on h-space activations.

    Process:
    1. Generate num_samples images, collecting h_t at each timestep
    2. For each timestep t, fit IncrementalPCA on flattened activations
    3. Stack per-timestep components into full directions (K, T, C, H, W)
    """

    def __init__(self, unet, scheduler, config: PipelineConfig):
        self.unet = unet
        self.scheduler = scheduler
        self.config = config
        self.device = next(unet.parameters()).device
        self.dtype = next(unet.parameters()).dtype

    def run(self) -> Dict:
        """
        Main entry: collect activations and fit PCA.

        Returns dict with:
        - 'components': Tensor (K, T, C, H, W) -- K principal directions
        - 'singular_values': Tensor (K, T)
        - 'explained_variance_ratio': Tensor (K, T)
        - 'mean_h': Tensor (T, C, H, W) -- mean activation per timestep
        - 'num_samples': int
        - 'num_inference_steps': int
        - 'seeds_used': list
        """
        disc_cfg = self.config.discovery
        samp_cfg = self.config.sampler
        num_samples = disc_cfg.num_samples
        num_steps = samp_cfg.num_inference_steps
        K = disc_cfg.num_components
        batch_size = disc_cfg.ipca_batch_size

        print(f"Stage 1: Collecting activations from {num_samples} samples "
              f"({num_steps} steps each)")

        # Initialize IncrementalPCA per timestep
        ipcas = [IncrementalPCA(n_components=K) for _ in range(num_steps)]

        # Accumulate activations in batches
        batch_buffers = [[] for _ in range(num_steps)]  # per-timestep lists
        seeds_used = []

        for i in tqdm(range(num_samples), desc="Generating samples"):
            seed = disc_cfg.seed_start + i
            seeds_used.append(seed)

            hook = BottleneckHook(self.unet, mode="capture")
            hook.register()
            try:
                _, h_acts = generate_with_seed(
                    self.unet, self.scheduler, hook,
                    seed=seed,
                    num_inference_steps=num_steps,
                    eta=samp_cfg.eta,
                    device=str(self.device),
                    dtype=self.dtype,
                )
            finally:
                hook.remove()

            # h_acts: (T, C, H, W) -- store flattened per timestep
            for t in range(num_steps):
                h_t = h_acts[t].numpy().flatten()  # (C*H*W,)
                batch_buffers[t].append(h_t)

            # When batch is full, partial_fit all timestep IPCAs
            if len(batch_buffers[0]) >= batch_size:
                for t in range(num_steps):
                    batch_matrix = np.stack(batch_buffers[t], axis=0)
                    ipcas[t].partial_fit(batch_matrix)
                    batch_buffers[t] = []

        # Flush remaining samples
        if len(batch_buffers[0]) > 0:
            for t in range(num_steps):
                batch_matrix = np.stack(batch_buffers[t], axis=0)
                # IncrementalPCA needs at least n_components samples per batch
                if batch_matrix.shape[0] >= K:
                    ipcas[t].partial_fit(batch_matrix)
                else:
                    print(f"Warning: Skipping final batch for t={t} "
                          f"(only {batch_matrix.shape[0]} samples, need {K})")

        # Extract components
        # Determine spatial shape from the first IPCA
        feature_dim = ipcas[0].components_.shape[1]
        C = 512
        H = W = int(np.sqrt(feature_dim // C))
        assert C * H * W == feature_dim, f"Cannot reshape {feature_dim} into (C, H, W)"

        components = torch.zeros(K, num_steps, C, H, W)
        singular_values = torch.zeros(K, num_steps)
        explained_variance_ratio = torch.zeros(K, num_steps)
        mean_h = torch.zeros(num_steps, C, H, W)

        for t in range(num_steps):
            comps = ipcas[t].components_  # (K, feature_dim)
            for k in range(K):
                components[k, t] = torch.from_numpy(
                    comps[k].reshape(C, H, W)
                ).float()
            singular_values[:, t] = torch.from_numpy(
                ipcas[t].singular_values_[:K]
            ).float()
            explained_variance_ratio[:, t] = torch.from_numpy(
                ipcas[t].explained_variance_ratio_[:K]
            ).float()
            mean_h[t] = torch.from_numpy(
                ipcas[t].mean_.reshape(C, H, W)
            ).float()

        result = {
            "components": components,
            "singular_values": singular_values,
            "explained_variance_ratio": explained_variance_ratio,
            "mean_h": mean_h,
            "num_samples": num_samples,
            "num_inference_steps": num_steps,
            "seeds_used": seeds_used,
        }

        # Save checkpoint
        output_path = os.path.join(self.config.output_dir, "stage1", "pca_components.pt")
        save_checkpoint(result, output_path)

        print(f"PCA complete. Top component explains "
              f"{explained_variance_ratio[0].mean():.4f} of variance (avg over timesteps)")

        return result
