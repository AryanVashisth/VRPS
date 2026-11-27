"""ONNX-only crop recommendation inference (Android-ready shape)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Union

import numpy as np
import onnxruntime as ort

_ROOT = Path(__file__).resolve().parents[1]
_ART = _ROOT / "artifacts"


class CropRecommendationModel:
    """Pilot-ready crop recommender using artifacts/schema + ONNX."""

    def __init__(self, artifacts_dir: Path | None = None) -> None:
        art = Path(artifacts_dir) if artifacts_dir else _ART
        self.schema = json.loads((art / "schema.json").read_text(encoding="utf-8"))
        self.labels = {
            int(k): v for k, v in json.loads((art / "labels.json").read_text(encoding="utf-8")).items()
        }
        self.feature_order: List[str] = list(self.schema["feature_order"])
        model_name = self.schema.get("android", {}).get("model_file", "crop_recommendation_v1.onnx")
        self.session = ort.InferenceSession(
            str(art / model_name),
            providers=["CPUExecutionProvider"],
        )
        self._input = self.session.get_inputs()[0].name

    def _vector(self, features: Mapping[str, float]) -> np.ndarray:
        missing = [f for f in self.feature_order if f not in features]
        if missing:
            raise KeyError(f"Missing features: {missing}")
        row = [float(features[f]) if features[f] is not None else np.nan for f in self.feature_order]
        return np.asarray([row], dtype=np.float32)

    def predict_one(
        self, features: Mapping[str, float]
    ) -> Dict[str, Union[int, str, float, Dict[str, float]]]:
        x = self._vector(features)
        outs = self.session.run(None, {self._input: x})
        label_id = int(np.asarray(outs[0]).reshape(-1)[0])
        probs_raw = np.asarray(outs[1])
        probs = probs_raw[0] if probs_raw.ndim == 2 else probs_raw
        prob_map = {self.labels[i]: float(probs[i]) for i in range(len(self.labels))}
        ranked = sorted(prob_map.items(), key=lambda item: item[1], reverse=True)
        return {
            "crop_id": label_id,
            "crop": self.labels[label_id],
            "confidence": float(prob_map[self.labels[label_id]]),
            "probabilities": dict(ranked),
            "top3": [{"crop": name, "probability": prob} for name, prob in ranked[:3]],
            "model_version": self.schema.get("version"),
        }

    def predict_many(self, rows: Sequence[Mapping[str, float]]) -> List[Dict]:
        return [self.predict_one(row) for row in rows]


def demo() -> None:
    model = CropRecommendationModel()
    sample = {
        "N": 90,
        "P": 42,
        "K": 43,
        "pH": 6.5,
        "Temperature": 20.9,
        "Humidity": 82.0,
        "Rainfall": 202.9,
    }
    print(json.dumps(model.predict_one(sample), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend a crop from soil/weather features.")
    parser.add_argument("--n", type=float)
    parser.add_argument("--p", type=float)
    parser.add_argument("--k", type=float)
    parser.add_argument("--ph", type=float)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--humidity", type=float)
    parser.add_argument("--rainfall", type=float)
    parser.add_argument("--artifacts", type=Path, default=_ART)
    args = parser.parse_args()

    values = {
        "N": args.n,
        "P": args.p,
        "K": args.k,
        "pH": args.ph,
        "Temperature": args.temperature,
        "Humidity": args.humidity,
        "Rainfall": args.rainfall,
    }
    if any(v is None for v in values.values()):
        demo()
        return

    model = CropRecommendationModel(args.artifacts)
    print(json.dumps(model.predict_one(values), indent=2))


if __name__ == "__main__":
    main()
