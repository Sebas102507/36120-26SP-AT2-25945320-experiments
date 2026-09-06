import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "36120_26sp_at2_25945320_experiments"))

from modeling.deployment import promote_bundle


def test_promote_bundle_copies_only_manifest_referenced_artifacts(tmp_path):
    bundle = tmp_path / "experiment"
    production = tmp_path / "production"
    bundle.mkdir()
    production.mkdir()
    (bundle / "model.joblib").write_bytes(b"model")
    (bundle / "metadata.json").write_text("{}")
    (bundle / "cci_model_manifest.json").write_text(
        json.dumps(
            {
                "adapter": "joblib",
                "model_files": ["model.joblib"],
                "metadata_file": "metadata.json",
            }
        )
    )
    (production / "cci_stale.joblib").write_bytes(b"stale")
    (production / ".gitkeep").touch()

    promote_bundle(bundle, production)

    assert sorted(path.name for path in production.iterdir()) == [
        ".gitkeep",
        "cci_model_manifest.json",
        "metadata.json",
        "model.joblib",
    ]
