from __future__ import annotations

from pathlib import Path

import typer

from .config import load_config
from .infer import run_inference
from .train import train_pipeline

app = typer.Typer(help="Bottle cap sorter (bsort) CLI.")


@app.command()
def train(config: str = typer.Option(..., help="Path to settings YAML config.")) -> None:
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


if __name__ == "__main__":
    app()
