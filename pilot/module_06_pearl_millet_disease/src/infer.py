"""ONNX inference for Module 6 pearl millet leaf disease (colour features).

ONNX input is 30 colour features, not pixels. predict_from_image uses
src.features.colour_features to compute them first.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Sequence, Union

import numpy as np
import onnxruntime as ort

try:
    from .features import colour_features
except ImportError:  # loaded as a loose file by the pilot console
    from features import colour_features

_ROOT = Path(__file__).resolve().parents[1]
_ART = _ROOT / "artifacts"


class PearlMilletDiseaseModel:
    def __init__(self, artifacts_dir: Path | None = None) -> None:
        art = Path(artifacts_dir) if artifacts_dir else _ART
        self.schema = json.loads((art / "schema.json").read_text())
        self.labels = {int(k): v for k, v in json.loads((art / "labels.json").read_text()).items()}
        self.feature_order = list(self.schema["feature_order"])
        self.session = ort.InferenceSession(str(art / "disease_v1.onnx"), providers=["CPUExecutionProvider"])
        self._input = self.session.get_inputs()[0].name

    def _vector(self, features: Union[Mapping[str, float], Sequence[float], np.ndarray]) -> np.ndarray:
        if isinstance(features, Mapping):
            missing = [f for f in self.feature_order if f not in features]
            if missing:
                raise KeyError(f"Missing features: {missing}")
            row = [np.nan if features[f] is None else float(features[f]) for f in self.feature_order]
        else:
            row = np.asarray(features, dtype=np.float32).reshape(-1).tolist()
            if len(row) != len(self.feature_order):
                raise ValueError(f"Expected {len(self.feature_order)} features, got {len(row)}")
        return np.asarray([row], dtype=np.float32)

    def predict_from_features(self, features) -> Dict:
        outs = self.session.run(None, {self._input: self._vector(features)})
        label = int(np.asarray(outs[0]).reshape(-1)[0])
        probs = np.asarray(outs[1])
        probs = probs[0] if probs.ndim == 2 else probs
        prob_map = {self.labels[i]: float(probs[i]) for i in range(len(self.labels))}
        return {
            "disease_id": label,
            "disease": self.labels[label],
            "confidence": float(max(prob_map.values())),
            "probabilities": prob_map,
            "model_version": self.schema["version"],
        }

    def predict_from_image(self, path) -> Dict:
        return self.predict_from_features(colour_features(path))


_default: PearlMilletDiseaseModel | None = None


def _model() -> PearlMilletDiseaseModel:
    global _default
    if _default is None:
        _default = PearlMilletDiseaseModel()
    return _default


def predict_from_features(features) -> Dict:
    return _model().predict_from_features(features)


def predict_from_image(path) -> Dict:
    return _model().predict_from_image(path)


def demo() -> None:
    import sys
    demo_dir = _ROOT / "data" / "demo_images"
    path = sys.argv[1] if len(sys.argv) > 1 else sorted(demo_dir.glob("*"))[0]
    print(path)
    print(json.dumps(predict_from_image(path), indent=2))


if __name__ == "__main__":
    demo()
