from __future__ import annotations

import time
from pathlib import Path

import cv2
import typer
from ultralytics import YOLO

from .config import load_config
from .infer import run_inference
from .train import train_pipeline

app = typer.Typer(help="Bottle cap sorter (bsort) CLI.")


@app.command()
def train(
    config: str = typer.Option(..., help="Path to settings YAML config.")
) -> None:
    """Train the bottle cap detection model."""
    cfg = load_config(config)
    typer.echo("Starting training...")
    train_pipeline(cfg)
    typer.echo("Training finished.")


@app.command()
def infer(
    config: str = typer.Option(..., help="Path to settings YAML config."),
    image: str = typer.Option(..., "--image", "-i", help="Input image path."),
    output: str = typer.Option(
        None,
        "--output",
        "-o",
        help="Optional output image path.",
    ),
) -> None:
    """Run inference on a single image."""
    cfg = load_config(config)
    out_path = run_inference(cfg, image_path=image, output_path=output)
    typer.echo(f"Saved prediction image to: {out_path}")


@app.command()
def benchmark(
    config: str = typer.Option(..., help="Path to settings YAML config."),
    source: str = typer.Option(
        "0",
        "--source",
        "-s",
        help="Video source: '0' for webcam or path to video file.",
    ),
    frames: int = typer.Option(
        100,
        "--frames",
        "-n",
        min=1,
        help="Number of frames to measure.",
    ),
    imgsz: int | None = typer.Option(
        None,
        "--imgsz",
        help="Inference image size (override cfg.training.img_size).",
    ),
    weights: str | None = typer.Option(
        None,
        "--weights",
        "-w",
        help="Optional weights path; defaults to cfg.model.best_model_path.",
    ),
) -> None:
    """Benchmark inference latency (ms per frame) on video/webcam."""
    # pylint: disable=too-many-locals

    cfg = load_config(config)

    # Tentukan path weights
    weights_path = (
        Path(weights) if weights is not None else Path(cfg.model.best_model_path)
    )

    if not weights_path.exists():
        typer.echo(f"[ERROR] Weights not found at: {weights_path}")
        raise typer.Exit(code=1)

    # Tentukan ukuran gambar untuk inference
    img_size = imgsz if imgsz is not None else cfg.training.img_size

    typer.echo(f"Loading model from: {weights_path}")
    typer.echo(f"Using image size: {img_size}")
    typer.echo(f"Measuring latency on source: {source} (frames={frames})")

    model = YOLO(str(weights_path))

    # source bisa webcam (0) atau file video
    video_source: int | str
    if source == "0":
        video_source = 0
    else:
        video_source = source

    cap = cv2.VideoCapture(video_source)  # pylint: disable=no-member
    if not cap.isOpened():
        typer.echo(f"[ERROR] Cannot open video source: {source}")
        raise typer.Exit(code=1)

    warmup = 10
    times: list[float] = []

    # Warmup – biar model dan pipeline "panas" dulu
    typer.echo(f"Running {warmup} warmup frames...")
    for _ in range(warmup):
        ret, frame = cap.read()
        if not ret:
            break
        _ = model.predict(frame, imgsz=img_size, verbose=False)

    typer.echo("Measuring latency...")
    for _ in range(frames):
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        _ = model.predict(frame, imgsz=img_size, verbose=False)
        t1 = time.time()

        times.append((t1 - t0) * 1000.0)  # ms

    cap.release()

    if not times:
        typer.echo("[ERROR] No frames processed. Please check your video source.")
        raise typer.Exit(code=1)

    avg_ms = sum(times) / len(times)
    sorted_t = sorted(times)
    p50 = sorted_t[len(sorted_t) // 2]
    p95 = sorted_t[int(len(sorted_t) * 0.95) - 1]

    typer.echo("")
    typer.echo(f"Frames measured     : {len(times)}")
    typer.echo(f"Average latency     : {avg_ms:.2f} ms")
    typer.echo(f"Median latency (p50): {p50:.2f} ms")
    typer.echo(f"p95 latency         : {p95:.2f} ms")


if __name__ == "__main__":
    app()
