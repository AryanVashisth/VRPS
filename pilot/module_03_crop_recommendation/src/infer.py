"""On-device crop recommendation inference (ONNX Runtime)."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Mapping
import numpy as np
import onnxruntime as ort
_ART = Path(__file__).resolve().parents[1] / "artifacts"

class CropRecommendationModel:
    def __init__(self, artifacts_dir=None):
        art = Path(artifacts_dir) if artifacts_dir else _ART
        self.schema = json.loads((art/"schema.json").read_text())
        self.labels = {int(k): v for k,v in json.loads((art/"labels.json").read_text()).items()}
        self.feature_order = list(self.schema["feature_order"])
        self.session = ort.InferenceSession(str(art/self.schema["android"]["model_file"]), providers=["CPUExecutionProvider"])
        self._input = self.session.get_inputs()[0].name
    def predict_one(self, features: Mapping[str,float]) -> Dict:
        missing = [f for f in self.feature_order if f not in features]
        if missing: raise KeyError(missing)
        x = np.asarray([[float(features[f]) for f in self.feature_order]], dtype=np.float32)
        outs = self.session.run(None, {self._input: x})
        label = int(np.asarray(outs[0]).reshape(-1)[0])
        probs = np.asarray(outs[1])[0]
        prob_map = {self.labels[i]: float(probs[i]) for i in range(len(self.labels))}
        return {"crop_id": label, "crop": self.labels[label], "confidence": float(max(prob_map.values())),
                "probabilities_top3": dict(sorted(prob_map.items(), key=lambda kv: kv[1], reverse=True)[:3]),
                "model_version": self.schema["version"],
                "honesty": self.schema.get("honesty")}

def demo():
    m = CropRecommendationModel()
    sample = {"N":90,"P":42,"K":43,"temperature":20.8,"humidity":82.0,"ph":6.5,"rainfall":202.9}
    print(json.dumps(m.predict_one(sample), indent=2))
if __name__ == "__main__":
    demo()
