"""Create self-contained, API-compatible CCI experiment bundles."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

import joblib
import torch


def _write_json(path: Path, content: dict[str, Any]) -> None:
    path.write_text(json.dumps(content, indent=2), encoding="utf-8")


def export_sequence_bundle(
    *,
    output_dir: Path,
    experiment: str,
    model: torch.nn.Module,
    sequence_features: list[str],
    lookback_days: int,
    input_scaler: Any,
    target_scaler: Any,
    metadata: dict[str, Any],
    model_family: str,
) -> Path:
    """Export an LSTM/GRU model as TorchScript and write its API manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_file = f"{experiment}.torchscript.pt"
    model.eval()
    example = torch.zeros(1, lookback_days, len(sequence_features))
    torch.jit.trace(model, example).save(output_dir / model_file)
    input_scaler_file = f"{experiment}_input_scaler.joblib"
    target_scaler_file = f"{experiment}_target_scaler.joblib"
    metadata_file = f"{experiment}_metadata.json"
    joblib.dump(input_scaler, output_dir / input_scaler_file)
    joblib.dump(target_scaler, output_dir / target_scaler_file)
    _write_json(output_dir / metadata_file, metadata)
    _write_json(
        output_dir / "cci_model_manifest.json",
        {
            "version": 1,
            "target": "comfort_climate",
            "experiment": experiment,
            "model_family": model_family,
            "adapter": "torchscript",
            "input_kind": "sequence",
            "model_file": model_file,
            "input_scaler_file": input_scaler_file,
            "target_scaler_file": target_scaler_file,
            "metadata_file": metadata_file,
            "sequence_features": sequence_features,
            "lookback_days": lookback_days,
            "horizons_days": [1, 2, 3],
            "target_inversion": {"type": "residual_plus_feature", "feature": "cci"},
        },
    )
    return output_dir


def export_tabular_bundle(
    *,
    output_dir: Path,
    experiment: str,
    models: dict[int, Any],
    feature_columns: list[str],
    metadata: dict[str, Any],
    model_family: str,
    history_days: int = 30,
) -> Path:
    """Export three horizon-specific sklearn/XGBoost models and a manifest."""
    if sorted(models) != [1, 2, 3]:
        raise ValueError("CCI tabular deployment requires models for horizons 1, 2, and 3.")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_files = []
    for horizon, model in sorted(models.items()):
        filename = f"{experiment}_h{horizon}.joblib"
        joblib.dump(model, output_dir / filename)
        model_files.append(filename)
    metadata_file = f"{experiment}_metadata.json"
    _write_json(output_dir / metadata_file, metadata)
    _write_json(
        output_dir / "cci_model_manifest.json",
        {
            "version": 1,
            "target": "comfort_climate",
            "experiment": experiment,
            "model_family": model_family,
            "adapter": "joblib",
            "input_kind": "tabular",
            "model_files": model_files,
            "metadata_file": metadata_file,
            "feature_columns": feature_columns,
            "history_days": history_days,
            "lookback_days": history_days,
            "horizons_days": [1, 2, 3],
            "target_inversion": {"type": "none"},
        },
    )
    return output_dir


def promote_bundle(bundle_dir: Path, production_dir: Path) -> Path:
    """Replace production with one validated experiment bundle."""
    manifest_path = bundle_dir / "cci_model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_names = [
        *(manifest.get("model_files") or [manifest.get("model_file")]),
        manifest.get("input_scaler_file"),
        manifest.get("target_scaler_file"),
        manifest.get("metadata_file"),
        "cci_model_manifest.json",
    ]
    artifact_names = [name for name in artifact_names if name]
    missing = [name for name in artifact_names if not (bundle_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Bundle is incomplete: {', '.join(missing)}")

    production_dir.mkdir(parents=True, exist_ok=True)
    for path in production_dir.iterdir():
        if path.name != ".gitkeep":
            path.unlink()
    for name in artifact_names:
        shutil.copy2(bundle_dir / name, production_dir / name)
    return production_dir
