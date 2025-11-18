from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class DatasetConfig:
    train_images: str
    train_labels: str
    val_images: str
    val_labels: str
    num_classes: int


@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    img_size: int
    learning_rate: float
    device: str


@dataclass
class ModelConfig:
    name: str
    pretrained_weights: str
    output_dir: str
    best_model_path: str


@dataclass
class InferenceConfig:
    conf_threshold: float
    iou_threshold: float
    output_dir: str


@dataclass
class WandbConfig:
    enabled: bool
    project: str
    entity: str


@dataclass
class AppConfig:
    dataset: DatasetConfig
    training: TrainingConfig
    model: ModelConfig
    inference: InferenceConfig
    wandb: WandbConfig


def load_config(path: str | Path) -> AppConfig:
    """Load application configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Parsed application configuration as an AppConfig instance.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    dataset = DatasetConfig(**raw["dataset"])
    training = TrainingConfig(**raw["training"])
    model = ModelConfig(**raw["model"])
    inference = InferenceConfig(**raw["inference"])
    wandb_cfg = WandbConfig(**raw["wandb"])

    return AppConfig(
        dataset=dataset,
        training=training,
        model=model,
        inference=inference,
        wandb=wandb_cfg,
    )
