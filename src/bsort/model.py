from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics import YOLO


class BottleCapModel:
    """Wrapper around a YOLO model for bottle cap detection."""

    def __init__(self, model_name: str, pretrained_weights: str) -> None:
        """Initialize the YOLO model.

        Args:
            model_name: Model architecture name (e.g. 'yolov8n').
            pretrained_weights: Path or alias to pretrained weights file.
        """
        # Bisa langsung pakai weights, nama cuma buat dokumentasi
        self.model_name = model_name
        self.model = YOLO(pretrained_weights)

    def train(
        self,
        data_yaml_path: str | Path,
        epochs: int,
        batch_size: int,
        img_size: int,
        learning_rate: float,
        device: str,
        project_dir: str | Path,
        run_name: str = "bsort_train",
    ) -> Any:
        """Train the YOLO model using Ultralytics training API.

        Args:
            data_yaml_path: Path to a YOLO-format data.yaml file.
            epochs: Number of training epochs.
            batch_size: Batch size.
            img_size: Image size (square).
            learning_rate: Learning rate.
            device: Device to use ('cpu' or 'cuda').
            project_dir: Directory to store training artifacts.
            run_name: Name of the training run.

        Returns:
            Training results object from Ultralytics.
        """
        results = self.model.train(
            data=str(data_yaml_path),
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            lr0=learning_rate,
            device=device,
            project=str(project_dir),
            name=run_name,
        )
        return results

    def save(self, path: str | Path) -> None:
        """Save model weights.

        Args:
            path: Output path for the weights file.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))

    @classmethod
    def load(cls, weights_path: str | Path) -> "BottleCapModel":
        """Load model from weights.

        Args:
            weights_path: Path to the YOLO weights file.

        Returns:
            BottleCapModel instance.
        """
        model = cls(model_name="custom", pretrained_weights=str(weights_path))
        return model
