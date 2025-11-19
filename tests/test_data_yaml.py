from pathlib import Path

import yaml

from bsort.config import AppConfig, load_config
from bsort.train import build_yolo_data_yaml


def test_build_yolo_data_yaml_creates_valid_file(tmp_path: Path) -> None:
    """build_yolo_data_yaml should create a YAML file with train/val/names keys."""
    cfg: AppConfig = load_config(Path("settings.yaml"))

    out_path = tmp_path / "data_bottlecap.yaml"
    generated_path = build_yolo_data_yaml(cfg, out_path)

    assert generated_path.exists(), "Data YAML file should be created"

    with generated_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert "train" in data
    assert "val" in data
    assert "names" in data
    assert isinstance(data["names"], list)
    assert len(data["names"]) == cfg.dataset.num_classes
