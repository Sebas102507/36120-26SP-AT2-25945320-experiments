"""Export CCI LSTM or GRU state-dictionary checkpoints to portable TorchScript."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import typer

app = typer.Typer()


class _SequenceForecastModel(nn.Module):
    def __init__(
        self, encoder: type[nn.LSTM | nn.GRU], n_features: int, hidden_size: int,
        dropout: float, n_horizons: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder(n_features, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, n_horizons)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output, _ = self.encoder(inputs)
        return self.head(self.dropout(output[:, -1, :]))


@app.command()
def main(checkpoint_path: Path, output_path: Path) -> None:
    """Convert an experiment checkpoint into a graph the API can load directly."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    architecture = checkpoint["architecture"].upper()
    encoders: dict[str, type[nn.LSTM | nn.GRU]] = {"LSTM": nn.LSTM, "GRU": nn.GRU}
    if architecture not in encoders:
        raise typer.BadParameter(f"Unsupported architecture: {architecture}")

    model = _SequenceForecastModel(
        encoders[architecture],
        checkpoint["n_features"],
        checkpoint["hidden_size"],
        checkpoint["dropout"],
        checkpoint["n_horizons"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    example = torch.zeros(1, checkpoint["lookback"], checkpoint["n_features"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.trace(model, example).save(output_path)
    typer.echo(f"Saved TorchScript model to {output_path}")


if __name__ == "__main__":
    app()
