"""Explicitly promote one API-compatible CCI experiment bundle."""

from __future__ import annotations

from pathlib import Path
import sys

import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modeling.deployment import promote_bundle

app = typer.Typer()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = PROJECT_ROOT / "models" / "comfort_climate" / "experiments"
PRODUCTION_DIR = PROJECT_ROOT / "models" / "comfort_climate" / "production"


@app.command()
def main(experiment: str) -> None:
    """Promote one validated experiment without selecting it automatically."""
    destination = promote_bundle(EXPERIMENTS_DIR / experiment, PRODUCTION_DIR)
    typer.echo(f"Promoted {experiment} to {destination}")


if __name__ == "__main__":
    app()
