from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

import wandb

from .config import AppConfig
from .model import BottleCapModel


def build_yolo_data_yaml(cfg: AppConfig, output_path: str | Path) -> Path:
    """Create YOLO data.yaml file from config.

    Args:
        cfg: Application configuration.
        output_path: Destination path for the YAML file.

    Returns:
        Path to the created YAML file.
    """

    data = {
        "train": str(Path(cfg.dataset.train_images).resolve()),
        "val": str(Path(cfg.dataset.val_images).resolve()),
        "nc": cfg.dataset.num_classes,
        "names": ["light_blue", "dark_blue", "others"],
    }
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)
    return out_path


def train_pipeline(cfg: AppConfig, run_name: Optional[str] = None) -> None:
    """Run the full training pipeline.

    Args:
        cfg: Application configuration.
        run_name: Optional name for the training run.
    """
    project_dir = Path(cfg.model.output_dir)
    project_dir.mkdir(parents=True, exist_ok=True)

    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=run_name or "bsort_train",
            config={
                "epochs": cfg.training.epochs,
                "batch_size": cfg.training.batch_size,
                "img_size": cfg.training.img_size,
                "learning_rate": cfg.training.learning_rate,
            },
        )

    data_yaml = build_yolo_data_yaml(cfg, project_dir / "data_bottlecap.yaml")

    model = BottleCapModel(
        model_name=cfg.model.name,
        pretrained_weights=cfg.model.pretrained_weights,
    )

    results = model.train(
        data_yaml_path=data_yaml,
        epochs=cfg.training.epochs,
        batch_size=cfg.training.batch_size,
        img_size=cfg.training.img_size,
        learning_rate=cfg.training.learning_rate,
        device=cfg.training.device,
        project_dir=project_dir,
        run_name=run_name or "bsort_train",
    )

    exp_name = run_name or "bsort_train"
    best_weights_path = project_dir / exp_name / "weights" / "best.pt"

    if not best_weights_path.exists():
        raise FileNotFoundError(f"Best weights not found at: {best_weights_path}")

    dest_path = Path(cfg.model.best_model_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_bytes(best_weights_path.read_bytes())

    if cfg.wandb.enabled:
        wandb.finish()
