"""On-device-style inference for Module 5 crop yield (ONNX Runtime only)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import onnxruntime as ort

_ART = Path(__file__).resolve().parents[1] / "artifacts"


class CropYieldModel:
    def __init__(self, artifacts_dir: Path | None = None) -> None:
        art = Path(artifacts_dir) if artifacts_dir else _ART
        self.schema = json.loads((art / "schema.json").read_text())
        enc = self.schema["categorical_encoding"]
        self.maps = enc["maps"]
        self.unknown = enc["unknown_code"]
        self.session = ort.InferenceSession(
            str(art / self.schema["android"]["model_file"]), providers=["CPUExecutionProvider"]
        )

    def _vector(self, f: Mapping[str, object]) -> np.ndarray:
        row = []
        for name in self.schema["numeric_features"]:
            if name not in f:
                raise KeyError(f"Missing feature: {name}")
            row.append(float(f[name]))
        for name in self.schema["categorical_features"]:
            if name not in f:
                raise KeyError(f"Missing feature: {name}")
            row.append(float(self.maps[name].get(str(f[name]).strip(), self.unknown)))
        return np.asarray([row], dtype=np.float32)

    def predict_one(self, features: Mapping[str, object]) -> Dict[str, object]:
        y = self.session.run(None, {"input": self._vector(features)})[0]
        return {"yield_pred": float(np.asarray(y).reshape(-1)[0]), "model_version": self.schema["version"]}


def demo() -> None:
    m = CropYieldModel()
    base = {"Crop_Year": 2018, "Area": 5000, "Annual_Rainfall": 900.0, "Season": "Kharif", "State": "Maharashtra"}
    for crop in ("Bajra", "Rice", "Unknown Crop"):
        print(crop, json.dumps(m.predict_one({**base, "Crop": crop})))


if __name__ == "__main__":
    demo()
