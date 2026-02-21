"""Stage 3: Automated labeling using CLIP and VLM approaches."""

import os
from typing import Dict, List, Optional, Tuple
from collections import Counter

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.config import LabelingConfig
from src.utils import save_json, free_gpu_memory


# Stop words to ignore in VLM caption differencing
_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "of", "in", "on", "at",
    "to", "for", "and", "or", "but", "with", "by", "from", "this", "that",
    "it", "its", "his", "her", "their", "my", "your",
}


class CLIPLabeler:
    """
    Approach A: CLIP zero-shot classification of semantic changes.

    For each direction, computes:
    delta_S(attr) = mean over seeds of [CLIP(I_pos, text) - CLIP(I_neg, text)]

    Uses logit-scaled cosine similarity for more meaningful scores.
    Also uses paired attribute detection (e.g., "smiling" vs "frowning")
    to identify bidirectional changes.
    """

    def __init__(self, config: LabelingConfig, device: str):
        self.config = config
        self.device = device
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.logit_scale = 1.0

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
        # Extract learned logit scale for amplifying cosine similarities
        self.logit_scale = self.model.logit_scale.exp().item()
        print(f"CLIP model loaded on {self.device} (logit_scale={self.logit_scale:.1f})")

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

        Uses logit-scaled similarities and selects the attribute with
        highest absolute delta (detecting both positive and negative changes).
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
            std_delta = np.std(all_deltas, axis=0)

            # Select by highest absolute delta (detects both + and - changes)
            abs_delta = np.abs(mean_delta)
            top_idx = int(np.argmax(abs_delta))

            # Also find top positive and top negative labels
            top_pos_idx = int(np.argmax(mean_delta))
            top_neg_idx = int(np.argmin(mean_delta))

            # Ranked list (top-3 by absolute delta)
            ranked_indices = np.argsort(abs_delta)[::-1]
            top3 = [
                {"label": attributes[int(i)], "score": float(mean_delta[int(i)])}
                for i in ranked_indices[:3]
            ]

            # Specific label: exclude overly broad gender attributes
            broad_attrs = {"a male person", "a female person"}
            specific_idx = top_idx
            for i in ranked_indices:
                if attributes[int(i)] not in broad_attrs:
                    specific_idx = int(i)
                    break

            results[f"direction_{k}"] = {
                "scores": {attr: float(mean_delta[i])
                           for i, attr in enumerate(attributes)},
                "scores_std": {attr: float(std_delta[i])
                               for i, attr in enumerate(attributes)},
                "top_label": attributes[top_idx],
                "top_score": float(mean_delta[top_idx]),
                "top_abs_score": float(abs_delta[top_idx]),
                "specific_label": attributes[specific_idx],
                "specific_score": float(mean_delta[specific_idx]),
                "top3": top3,
                "top_positive": {
                    "label": attributes[top_pos_idx],
                    "score": float(mean_delta[top_pos_idx]),
                },
                "top_negative": {
                    "label": attributes[top_neg_idx],
                    "score": float(mean_delta[top_neg_idx]),
                },
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
        Compute logit-scaled CLIP similarity delta.
        Returns: logit_scale * (cos_sim(pos, text) - cos_sim(neg, text))
        """
        img_pos_tensor = self.preprocess(img_pos).unsqueeze(0).to(self.device).half()
        img_neg_tensor = self.preprocess(img_neg).unsqueeze(0).to(self.device).half()

        pos_features = self.model.encode_image(img_pos_tensor)
        neg_features = self.model.encode_image(img_neg_tensor)
        pos_features = pos_features / pos_features.norm(dim=-1, keepdim=True)
        neg_features = neg_features / neg_features.norm(dim=-1, keepdim=True)

        # Apply logit scale for more meaningful scores
        pos_sim = (self.logit_scale * pos_features @ text_features.T).squeeze(0).cpu().numpy()
        neg_sim = (self.logit_scale * neg_features @ text_features.T).squeeze(0).cpu().numpy()

        return pos_sim - neg_sim


class VLMLabeler:
    """
    Approach B: VLM difference captioning using BLIP-2.

    Strategy: Caption positive and negative images with targeted VQA prompts,
    then analyze differences across captions to identify the semantic change.
    """

    # VQA prompts targeting specific facial attributes
    VQA_PROMPTS = [
        "Question: Describe this person's appearance briefly. Answer:",
        "Question: What stands out about this person's face? Answer:",
        "Question: Describe the hair, expression, and accessories. Answer:",
    ]

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
        For each direction, use multiple VQA prompts on positive and negative
        images, then analyze differences to determine the semantic change.
        """
        edits_by_dir = {}
        for e in edit_metadata["edits"]:
            k = e["direction_idx"]
            if k not in edits_by_dir:
                edits_by_dir[k] = []
            edits_by_dir[k].append(e)

        results = {}

        for k in tqdm(sorted(edits_by_dir.keys()), desc="VLM captioning"):
            all_captions = []  # list of (neg_caption, pos_caption) per seed

            for e in edits_by_dir[k]:
                img_pos = Image.open(os.path.join(output_dir, e["pos_path"]))
                img_neg = Image.open(os.path.join(output_dir, e["neg_path"]))

                # Use multiple prompts and collect all captions
                neg_caps = []
                pos_caps = []
                for prompt in self.VQA_PROMPTS:
                    neg_caps.append(self._caption_vqa(img_neg, prompt))
                    pos_caps.append(self._caption_vqa(img_pos, prompt))

                neg_combined = " | ".join(neg_caps)
                pos_combined = " | ".join(pos_caps)
                all_captions.append((neg_combined, pos_combined))

            # Analyze differences
            consensus = self._analyze_differences(all_captions)
            caption_strings = [f"{neg} -> {pos}" for neg, pos in all_captions]

            results[f"direction_{k}"] = {
                "captions": caption_strings,
                "neg_captions": [neg for neg, _ in all_captions],
                "pos_captions": [pos for _, pos in all_captions],
                "consensus_label": consensus,
            }

        return results

    @torch.no_grad()
    def _caption_vqa(self, image: Image.Image, prompt: str) -> str:
        """Run VQA on a single image with a specific prompt."""
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.model.device) if hasattr(v, 'to') else v
                  for k, v in inputs.items()}
        if self.model.device.type != "cpu":
            inputs = {k: v.half() if hasattr(v, 'half') and v.dtype == torch.float32 else v
                      for k, v in inputs.items()}

        output_ids = self.model.generate(**inputs, max_new_tokens=40)
        # Decode only newly generated tokens (skip the input prompt tokens)
        input_len = inputs.get("input_ids", torch.empty(0)).shape[-1] if "input_ids" in inputs else 0
        if input_len > 0 and output_ids.shape[1] > input_len:
            new_tokens = output_ids[0, input_len:]
        else:
            new_tokens = output_ids[0]
        caption = self.processor.decode(new_tokens, skip_special_tokens=True).strip()
        return caption

    def _analyze_differences(
        self, all_captions: List[Tuple[str, str]]
    ) -> str:
        """
        Analyze caption pairs to find consistent differences.
        Returns a human-readable summary of the change.
        """
        words_gained = Counter()  # words appearing more in pos
        words_lost = Counter()    # words appearing more in neg

        for neg_text, pos_text in all_captions:
            neg_words = self._extract_content_words(neg_text)
            pos_words = self._extract_content_words(pos_text)

            for w in pos_words - neg_words:
                words_gained[w] += 1
            for w in neg_words - pos_words:
                words_lost[w] += 1

        # Filter to words appearing in at least 2 seed comparisons
        min_count = max(1, len(all_captions) // 3)
        gained = [(w, c) for w, c in words_gained.most_common(10) if c >= min_count]
        lost = [(w, c) for w, c in words_lost.most_common(10) if c >= min_count]

        parts = []
        if gained:
            parts.append("+" + ", ".join(w for w, _ in gained[:5]))
        if lost:
            parts.append("-" + ", ".join(w for w, _ in lost[:5]))

        if parts:
            return " / ".join(parts)

        # Fallback: just note if captions are identical
        identical = sum(1 for n, p in all_captions if n.lower() == p.lower())
        if identical == len(all_captions):
            return "(no detectable difference)"
        return "(subtle difference)"

    @staticmethod
    def _extract_content_words(text: str) -> set:
        """Extract meaningful content words from caption text."""
        words = set()
        for w in text.lower().replace("|", " ").replace(",", " ").split():
            w = w.strip(".,!?;:'\"()[]")
            if w and w not in _STOP_WORDS and len(w) > 1:
                words.add(w)
        return words
