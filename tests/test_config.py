from pathlib import Path

from bsort.config import AppConfig, load_config


def test_load_config_return_type() -> None:
    """load_config should return an AppConfig instance."""
    cfg = load_config(Path("settings.yaml"))
    assert isinstance(cfg, AppConfig)
    assert cfg.dataset.train_images  # basic sanity
    assert cfg.dataset.num_classes > 0
