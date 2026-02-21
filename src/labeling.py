"""Stage 3: Automated labeling using CLIP and VLM approaches."""

import os
from typing import Dict, List, Optional

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.config import LabelingConfig
from src.utils import save_json, free_gpu_memory


class CLIPLabeler:
    """
    Approach A: CLIP zero-shot classification of semantic changes.

    For each direction, computes:
    delta_S(attr) = mean over seeds of [CLIP(I_pos, text) - CLIP(I_neg, text)]

    Assigns the attribute with highest positive delta_S.
    """

    def __init__(self, config: LabelingConfig, device: str):
        self.config = config
        self.device = device
        self.model = None
        self.preprocess = None
        self.tokenizer = None

    def load_model(self):
        """Load CLIP model (open_clip) in fp16."""
        import open_clip

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.config.clip_model_name,
            pretrained=self.config.clip_pretrained,
            device=self.device,
        )
        self.tokenizer = open_clip.get_tokenizer(self.config.clip_model_name)
        self.model = self.model.half()
        self.model.eval()
        print(f"CLIP model loaded on {self.device}")

    def unload_model(self):
        """Unload CLIP model from GPU."""
        del self.model
        del self.preprocess
        del self.tokenizer
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        free_gpu_memory()
        print("CLIP model unloaded")

    def label_directions(self, edit_metadata: Dict, output_dir: str) -> Dict:
        """
        For each direction, compute CLIP similarity delta for all attributes.

        Args:
            edit_metadata: Stage 2 output with 'edits' list
            output_dir: Base output directory

        Returns:
            Dict mapping direction index to scores and labels.
        """
        attributes = self.config.attribute_list

        # Pre-encode text attributes
        text_tokens = self.tokenizer(attributes).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Group edits by direction
        edits_by_dir = {}
        for e in edit_metadata["edits"]:
            k = e["direction_idx"]
            if k not in edits_by_dir:
                edits_by_dir[k] = []
            edits_by_dir[k].append(e)

        results = {}

        for k in tqdm(sorted(edits_by_dir.keys()), desc="CLIP labeling"):
            all_deltas = []

            for e in edits_by_dir[k]:
                img_pos = Image.open(os.path.join(output_dir, e["pos_path"]))
                img_neg = Image.open(os.path.join(output_dir, e["neg_path"]))

                delta = self._compute_clip_delta(img_pos, img_neg, text_features)
                all_deltas.append(delta)

            # Average across seeds
            mean_delta = np.mean(all_deltas, axis=0)
            top_idx = int(np.argmax(mean_delta))

            results[f"direction_{k}"] = {
                "scores": {attr: float(mean_delta[i])
                           for i, attr in enumerate(attributes)},
                "top_label": attributes[top_idx],
                "top_score": float(mean_delta[top_idx]),
                "per_seed_deltas": [
                    {attr: float(d[i]) for i, attr in enumerate(attributes)}
                    for d in all_deltas
                ],
            }

        return results

    @torch.no_grad()
    def _compute_clip_delta(
        self,
        img_pos: Image.Image,
        img_neg: Image.Image,
        text_features: torch.Tensor,
    ) -> np.ndarray:
        """
        Compute CLIP(I_pos, text) - CLIP(I_neg, text) for each text.
        Returns array of cosine similarity deltas.
        """
        img_pos_tensor = self.preprocess(img_pos).unsqueeze(0).to(self.device).half()
        img_neg_tensor = self.preprocess(img_neg).unsqueeze(0).to(self.device).half()

        pos_features = self.model.encode_image(img_pos_tensor)
        neg_features = self.model.encode_image(img_neg_tensor)
        pos_features = pos_features / pos_features.norm(dim=-1, keepdim=True)
        neg_features = neg_features / neg_features.norm(dim=-1, keepdim=True)

        pos_sim = (pos_features @ text_features.T).squeeze(0).cpu().numpy()
        neg_sim = (neg_features @ text_features.T).squeeze(0).cpu().numpy()

        return pos_sim - neg_sim


class VLMLabeler:
    """
    Approach B: VLM difference captioning.

    Creates a side-by-side composite of I_neg and I_pos,
    then prompts a VLM to describe the difference.
    """

    def __init__(self, config: LabelingConfig, device: str):
        self.config = config
        self.device = device
        self.model = None
        self.processor = None

    def load_model(self):
        """Load BLIP-2 model. Falls back to CPU if GPU memory insufficient."""
        from transformers import Blip2Processor, Blip2ForConditionalGeneration

        print(f"Loading VLM: {self.config.vlm_model_name}...")
        self.processor = Blip2Processor.from_pretrained(self.config.vlm_model_name)

        try:
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.config.vlm_model_name,
                torch_dtype=torch.float16,
            ).to(self.device)
            print(f"VLM loaded on {self.device}")
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                print(f"GPU OOM, falling back to CPU for VLM")
                free_gpu_memory()
                self.device = "cpu"
                self.model = Blip2ForConditionalGeneration.from_pretrained(
                    self.config.vlm_model_name,
                    torch_dtype=torch.float32,
                )
                print("VLM loaded on CPU")
            else:
                raise

        self.model.eval()

    def unload_model(self):
        """Unload VLM model."""
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        free_gpu_memory()
        print("VLM model unloaded")

    def caption_directions(self, edit_metadata: Dict, output_dir: str) -> Dict:
        """
        For each direction, create composite image and generate caption.

        Returns dict with captions per direction and seed.
        """
        # Group edits by direction
        edits_by_dir = {}
        for e in edit_metadata["edits"]:
            k = e["direction_idx"]
            if k not in edits_by_dir:
                edits_by_dir[k] = []
            edits_by_dir[k].append(e)

        results = {}

        for k in tqdm(sorted(edits_by_dir.keys()), desc="VLM captioning"):
            captions = []

            for e in edits_by_dir[k]:
                img_pos = Image.open(os.path.join(output_dir, e["pos_path"]))
                img_neg = Image.open(os.path.join(output_dir, e["neg_path"]))

                composite = self._create_composite(img_neg, img_pos)
                caption = self._generate_caption(composite)
                captions.append(caption)

            results[f"direction_{k}"] = {
                "captions": captions,
                "consensus_label": self._get_consensus(captions),
            }

        return results

    def _create_composite(
        self, img_neg: Image.Image, img_pos: Image.Image
    ) -> Image.Image:
        """Create side-by-side image for VLM input."""
        w, h = img_neg.size
        composite = Image.new("RGB", (w * 2 + 10, h), color=(255, 255, 255))
        composite.paste(img_neg, (0, 0))
        composite.paste(img_pos, (w + 10, 0))
        return composite

    @torch.no_grad()
    def _generate_caption(self, composite: Image.Image) -> str:
        """Run VLM inference on composite image."""
        inputs = self.processor(
            images=composite,
            text=self.config.vlm_prompt,
            return_tensors="pt",
        )
        # Move to model device
        inputs = {k: v.to(self.model.device) if hasattr(v, 'to') else v
                  for k, v in inputs.items()}
        if self.model.device.type != "cpu":
            inputs = {k: v.half() if hasattr(v, 'half') and v.dtype == torch.float32 else v
                      for k, v in inputs.items()}

        output_ids = self.model.generate(**inputs, max_new_tokens=20)
        caption = self.processor.decode(output_ids[0], skip_special_tokens=True).strip()
        return caption

    def _get_consensus(self, captions: List[str]) -> str:
        """Get most common caption as consensus label."""
        from collections import Counter
        # Normalize captions
        normalized = [c.lower().strip().rstrip(".") for c in captions]
        if not normalized:
            return ""
        counter = Counter(normalized)
        return counter.most_common(1)[0][0]
