"""On-device-style inference for Module 4 fertilizer recommendation (ONNX Runtime only)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import onnxruntime as ort

_ART = Path(__file__).resolve().parents[1] / "artifacts"


class FertilizerModel:
    def __init__(self, artifacts_dir: Path | None = None) -> None:
        art = Path(artifacts_dir) if artifacts_dir else _ART
        self.schema = json.loads((art / "schema.json").read_text())
        self.labels = {int(k): v for k, v in json.loads((art / "labels.json").read_text()).items()}
        self.inputs = self.schema["inputs"]
        self.session = ort.InferenceSession(
            str(art / self.schema["android"]["model_file"]), providers=["CPUExecutionProvider"]
        )

    def predict_one(self, features: Mapping[str, object]) -> Dict[str, object]:
        feed = {}
        for spec in self.inputs:
            name = spec["name"]
            if name not in features:
                raise KeyError(f"Missing feature: {name}")
            if spec["dtype"] == "string":
                feed[spec["onnx_input"]] = np.array([[str(features[name]).strip()]], dtype=object)
            else:
                v = features[name]
                feed[spec["onnx_input"]] = np.array([[np.nan if v is None else float(v)]], dtype=np.float32)
        label, probs = self.session.run(None, feed)
        fid = int(np.asarray(label).reshape(-1)[0])
        p = np.asarray(probs).reshape(-1)
        prob_map = {self.labels[i]: float(p[i]) for i in range(len(self.labels))}
        return {
            "fertilizer_id": fid,
            "fertilizer": self.labels[fid],
            "confidence": float(p[fid]),
            "probabilities": prob_map,
            "model_version": self.schema["version"],
        }


def demo() -> None:
    sample = {
        "Temperature": 29, "Humidity": 52, "Moisture": 45, "Nitrogen": 12,
        "Potassium": 0, "Phosphorous": 36, "Soil Type": "Loamy", "Crop Type": "Sugarcane",
    }
    print(json.dumps(FertilizerModel().predict_one(sample), indent=2))


if __name__ == "__main__":
    demo()
