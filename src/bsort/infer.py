from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np

from .config import AppConfig
from .model import BottleCapModel


def draw_boxes(
    image: np.ndarray,
    boxes: List[List[float]],
    scores: List[float],
    class_ids: List[int],
    class_names: List[str],
) -> np.ndarray:
    """Draw bounding boxes on an image.

    Args:
        image: Input image array in BGR format.
        boxes: List of bounding boxes [x1, y1, x2, y2].
        scores: Detection scores.
        class_ids: Detected class ids.
        class_names: Class names.

    Returns:
        Image with bounding boxes drawn.
    """
    img = image.copy()
    for box, score, cls_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)
        label = f"{class_names[cls_id]} {score:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            label,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return img


def run_inference(cfg: AppConfig, image_path: str, output_path: str | None = None) -> str:
    """Run inference on a single image and save the result.

    Args:
        cfg: Application configuration.
        image_path: Path to the input image.
        output_path: Optional path for the output image. If None, auto-generate.

    Returns:
        Path to the saved output image.
    """
    model = BottleCapModel.load(cfg.model.best_model_path)

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    results = model.model.predict(
        source=img,
        conf=cfg.inference.conf_threshold,
        iou=cfg.inference.iou_threshold,
        verbose=False,
    )[0]

    # ⬇️ TAMBAH DEBUG DI SINI
    if results.boxes is None or len(results.boxes) == 0:
        print("[INF] No boxes detected at current thresholds.")
    else:
        print(f"[INF] Detected {len(results.boxes)} boxes")
        for box in results.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            print(f" - cls={cls_id}, conf={conf:.2f}")

    boxes = results.boxes.xyxy.cpu().numpy().tolist() if results.boxes is not None else []
    scores = results.boxes.conf.cpu().numpy().tolist() if results.boxes is not None else []
    class_ids = results.boxes.cls.cpu().numpy().astype(int).tolist() if results.boxes is not None else []
    class_names = ["light_blue", "dark_blue", "others"]

    img_out = draw_boxes(img, boxes, scores, class_ids, class_names)

    out_dir = Path(cfg.inference.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        output_path = out_dir / (Path(image_path).stem + "_pred.jpg")
    else:
        output_path = Path(output_path)

    cv2.imwrite(str(output_path), img_out)
    return str(output_path)
