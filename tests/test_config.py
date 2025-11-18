from pathlib import Path

from bsort.config import load_config


def test_load_config(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        """
dataset:
  train_images: "data/train/images"
  train_labels: "data/train/labels"
  val_images: "data/val/images"
  val_labels: "data/val/labels"
  num_classes: 3
training:
  epochs: 10
  batch_size: 4
  img_size: 320
  learning_rate: 0.001
  device: "cpu"
model:
  name: "yolov8n"
  pretrained_weights: "yolov8n.pt"
  output_dir: "artifacts"
  best_model_path: "artifacts/best.pt"
inference:
  conf_threshold: 0.5
  iou_threshold: 0.45
  output_dir: "outputs"
wandb:
  enabled: false
  project: "test"
  entity: "test"
        """
    )
    cfg = load_config(cfg_file)
    assert cfg.dataset.num_classes == 3
    assert cfg.training.epochs == 10
