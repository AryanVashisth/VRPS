"""On-device-style inference for Module 2 soil moisture (4-hour ahead)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import onnxruntime as ort

_ART = Path(__file__).resolve().parents[1] / "artifacts"


class SoilMoistureModel:
    def __init__(self, artifacts_dir: Path | None = None) -> None:
        art = Path(artifacts_dir) if artifacts_dir else _ART
        self.schema = json.loads((art / "schema.json").read_text())
        self.feature_order = list(self.schema["feature_order"])
        self.session = ort.InferenceSession(
            str(art / self.schema["android"]["model_file"]),
            providers=["CPUExecutionProvider"],
        )
        self._input = self.session.get_inputs()[0].name

    def predict_one(self, features: Mapping[str, float]) -> Dict:
        missing = [f for f in self.feature_order if f not in features]
        if missing:
            raise KeyError(f"Missing features: {missing}")
        x = np.asarray([[float(features[f]) for f in self.feature_order]], dtype=np.float32)
        pred = float(np.asarray(self.session.run(None, {self._input: x})[0]).reshape(-1)[0])
        persistence = float(features["lag_0"])
        return {
            "water_content_4h": pred,
            "persistence_4h": persistence,
            "recommended": pred if self.schema.get("ship_model_beats_persistence") else persistence,
            "policy": self.schema.get("recommended_policy"),
            "model_version": self.schema["version"],
        }


def demo() -> None:
    model = SoilMoistureModel()
    sample = {
        "lag_0": 0.22, "lag_1": 0.22, "lag_4": 0.221, "lag_16": 0.218, "lag_96": 0.210,
        "mean_last_4h": 0.220, "std_last_4h": 0.002, "delta_1": 0.0, "delta_4": -0.001,
    }
    print(json.dumps(model.predict_one(sample), indent=2))


if __name__ == "__main__":
    demo()
