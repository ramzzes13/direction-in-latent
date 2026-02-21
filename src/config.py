"""Configuration system using dataclasses and YAML."""

from dataclasses import dataclass, field, asdict
from typing import List, Optional

import yaml


@dataclass
class ModelConfig:
    model_id: str = "google/ddpm-celebahq-256"
    torch_dtype: str = "float16"
    device: str = "auto"  # "auto" will pick GPU with most free memory


@dataclass
class SamplerConfig:
    num_inference_steps: int = 50
    eta: float = 1.0  # 1.0=DDPM stochastic, 0.0=DDIM deterministic


@dataclass
class DiscoveryConfig:
    num_samples: int = 500
    num_components: int = 20
    ipca_batch_size: int = 50
    seed_start: int = 0


@dataclass
class EditConfig:
    num_directions: int = 10
    num_seeds_per_direction: int = 5
    default_alpha: float = 5.0
    alpha_values: List[float] = field(default_factory=lambda: [3.0, 5.0, 7.0])
    edit_seeds: List[int] = field(default_factory=lambda: [42, 123, 256, 789, 1024])


@dataclass
class LabelingConfig:
    clip_model_name: str = "ViT-B-32"
    clip_pretrained: str = "openai"
    attribute_list: List[str] = field(default_factory=lambda: [
        "a face rotated to the left",
        "a face rotated to the right",
        "a smiling person",
        "a frowning person",
        "a male person",
        "a female person",
        "a person wearing glasses",
        "a person without glasses",
        "an older person",
        "a younger person",
        "a person with blonde hair",
        "a person with dark hair",
        "a bald person",
        "a person with long hair",
        "a person with bangs",
        "a person with a hat",
        "a person with eyes closed",
        "a person with eyes open",
        "a person with mouth open",
        "a person with mouth closed",
    ])
    vlm_model_name: str = "Salesforce/blip2-opt-2.7b"
    vlm_prompt: str = (
        "Question: What is the main visual difference between the left face "
        "and the right face? Answer in one or two words. Answer:"
    )


@dataclass
class PipelineConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    edit: EditConfig = field(default_factory=EditConfig)
    labeling: LabelingConfig = field(default_factory=LabelingConfig)
    output_dir: str = "outputs"


def load_config(yaml_path: Optional[str] = None) -> PipelineConfig:
    """Load config from YAML file, falling back to defaults."""
    config = PipelineConfig()
    if yaml_path is None:
        return config

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f) or {}

    # Merge YAML data into config
    if "model" in data:
        for k, v in data["model"].items():
            if hasattr(config.model, k):
                setattr(config.model, k, v)
    if "sampler" in data:
        for k, v in data["sampler"].items():
            if hasattr(config.sampler, k):
                setattr(config.sampler, k, v)
    if "discovery" in data:
        for k, v in data["discovery"].items():
            if hasattr(config.discovery, k):
                setattr(config.discovery, k, v)
    if "edit" in data:
        for k, v in data["edit"].items():
            if hasattr(config.edit, k):
                setattr(config.edit, k, v)
    if "labeling" in data:
        for k, v in data["labeling"].items():
            if hasattr(config.labeling, k):
                setattr(config.labeling, k, v)
    if "output_dir" in data:
        config.output_dir = data["output_dir"]

    return config


def save_config(config: PipelineConfig, path: str):
    """Save config to YAML for reproducibility."""
    data = asdict(config)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
