from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def yolo_to_xyxy(
    cx: float,
    cy: float,
    w: float,
    h: float,
    img_w: int,
    img_h: int,
) -> Tuple[int, int, int, int]:
    """Convert YOLO (cx, cy, w, h) normalized to pixel (x1, y1, x2, y2)."""
    x_center = cx * img_w
    y_center = cy * img_h
    box_w = w * img_w
    box_h = h * img_h

    x1 = int(x_center - box_w / 2)
    y1 = int(y_center - box_h / 2)
    x2 = int(x_center + box_w / 2)
    y2 = int(y_center + box_h / 2)

    # clamp to image size
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w - 1))
    y2 = max(0, min(y2, img_h - 1))

    return x1, y1, x2, y2


def decide_color_label(crop_bgr: np.ndarray) -> int:
    """Decide class id (0/1/2) based on average HSV of the crop.

    0 = light_blue
    1 = dark_blue
    2 = others
    """
    if crop_bgr.size == 0:
        return 2  # safety fallback

    crop_hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(crop_hsv)

    # ambil hanya area tengah, biar nggak kena background / frame
    h_center = h[h.shape[0]//4 : 3*h.shape[0]//4,
                h.shape[1]//4 : 3*h.shape[1]//4]
    s_center = s[...][h_center.shape[0]*0: h_center.shape[0]]
    v_center = v[...][h_center.shape[0]*0: h_center.shape[0]]

    h_mean = float(np.median(h_center))
    s_mean = float(np.median(s_center))
    v_mean = float(np.median(v_center))


    # Hue biru kira-kira di 80–140 (range ini bisa kamu tweak)
    # is_blue_hue = 95 <= h_mean <= 130
    is_blue_hue = 80 <= h_mean <= 140
    is_saturated = s_mean >= 50


    if is_blue_hue and is_saturated:
        if v_mean >= 100:
            return 0  # light_blue
        else:
            return 1  # dark_blue
    else:
        return 2  # others


def relabel_split(
    images_dir: Path,
    labels_dir: Path,
) -> None:
    """Relabel all YOLO labels in a split (train or val) based on color."""
    print(f"Relabeling split: images={images_dir}, labels={labels_dir}")

    label_files: List[Path] = sorted(labels_dir.glob("*.txt"))
    for label_path in label_files:
        image_path_jpg = images_dir / f"{label_path.stem}.jpg"
        image_path_png = images_dir / f"{label_path.stem}.png"

        if image_path_jpg.exists():
            image_path = image_path_jpg
        elif image_path_png.exists():
            image_path = image_path_png
        else:
            print(f"⚠ Image not found for label {label_path.name}, skipping")
            continue

        img = cv2.imread(str(image_path))
        if img is None:
            print(f"⚠ Failed to read image {image_path}, skipping")
            continue

        img_h, img_w = img.shape[:2]

        with label_path.open("r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]

        new_lines: List[str] = []
        for ln in lines:
            parts = ln.split()
            if len(parts) != 5:
                print(f"⚠ Invalid label line in {label_path}: {ln}")
                continue

            # YOLO: class_id cx cy w h
            _old_cls = int(float(parts[0]))
            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

            x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, w, h, img_w, img_h)
            crop = img[y1:y2, x1:x2]

            new_cls = decide_color_label(crop)

            # tulis lagi dalam format YOLO dengan class baru
            new_line = f"{new_cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
            new_lines.append(new_line)

        # overwrite file label dengan label baru
        with label_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + "\n")

        print(f"✅ Relabeled {label_path.name}")


def main() -> None:
    """Entry point: relabel train dan val."""
    base = Path("data")
    train_images = base / "train" / "images"
    train_labels = base / "train" / "labels"

    val_images = base / "val" / "images"
    val_labels = base / "val" / "labels"

    relabel_split(train_images, train_labels)
    relabel_split(val_images, val_labels)

    print("🎉 Done relabeling all splits.")


if __name__ == "__main__":
    main()
