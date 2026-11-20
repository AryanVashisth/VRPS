"""On-device-style inference for Module 1 soil fertility.

Uses ONNX Runtime only (Android: onnxruntime-android). Preprocessing is inside the ONNX graph.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Union

import numpy as np
import onnxruntime as ort

_ROOT = Path(__file__).resolve().parents[1]
_ART = _ROOT / "artifacts"


class SoilFertilityModel:
    """Pilot-ready soil fertility classifier."""

    def __init__(self, artifacts_dir: Path | None = None) -> None:
        art = Path(artifacts_dir) if artifacts_dir else _ART
        self.schema = json.loads((art / "schema.json").read_text())
        self.labels = {int(k): v for k, v in json.loads((art / "labels.json").read_text()).items()}
        self.feature_order: List[str] = list(self.schema["feature_order"])
        self.session = ort.InferenceSession(
            str(art / "soil_fertility_v1.onnx"),
            providers=["CPUExecutionProvider"],
        )
        self._input = self.session.get_inputs()[0].name

    def _vector(self, features: Mapping[str, float]) -> np.ndarray:
        missing = [f for f in self.feature_order if f not in features]
        if missing:
            raise KeyError(f"Missing features: {missing}")
        row = [float(features[f]) if features[f] is not None else np.nan for f in self.feature_order]
        return np.asarray([row], dtype=np.float32)

    def predict_one(self, features: Mapping[str, float]) -> Dict[str, Union[int, str, float, Dict[str, float]]]:
        x = self._vector(features)
        outs = self.session.run(None, {self._input: x})
        label = int(np.asarray(outs[0]).reshape(-1)[0])
        probs_raw = np.asarray(outs[1])
        if probs_raw.ndim == 2:
            probs = probs_raw[0]
        else:
            probs = probs_raw
        prob_map = {self.labels[i]: float(probs[i]) for i in range(len(self.labels))}
        return {
            "fertility_id": label,
            "fertility": self.labels[label],
            "confidence": float(max(prob_map.values())),
            "probabilities": prob_map,
            "model_version": self.schema["version"],
        }

    def predict_many(self, rows: Sequence[Mapping[str, float]]) -> List[Dict]:
        return [self.predict_one(r) for r in rows]


def demo() -> None:
    model = SoilFertilityModel()
    sample = {f: 0.0 for f in model.feature_order}
    # mid-range India-like soil
    sample.update({"N": 2, "P": 150, "K": 18, "ph": 7.2, "ec": 0.5, "oc": 0.55, "S": 10, "zn": 0.6, "fe": 5, "cu": 0.3, "Mn": 3, "B": 0.5})
    print(json.dumps(model.predict_one(sample), indent=2))


if __name__ == "__main__":
    demo()
