# Pilot Module 4: Fertilizer recommendation

Predicts a fertilizer name (7 classes) from soil, crop and weather readings.

Inference uses **ONNX Runtime only** (on Android: `onnxruntime-android`). No network is needed. Median imputation and one-hot encoding are inside the ONNX graph.

## Quick start

```bash
cd pilot/module_04_fertilizer_recommendation
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.infer
```

```python
from src.infer import FertilizerModel
FertilizerModel().predict_one({
    "Temperature": 29, "Humidity": 52, "Moisture": 45, "Nitrogen": 12,
    "Potassium": 0, "Phosphorous": 36, "Soil Type": "Loamy", "Crop Type": "Sugarcane",
})
# {fertilizer_id, fertilizer, confidence, probabilities, model_version}
```

## Ship to Android

Copy `artifacts/model.onnx`, `schema.json`, `labels.json`. Inputs are eight tensors named in `schema.json` (`onnx_input`): six `float32[1,1]` and two `string[1,1]` (Soil_Type, Crop_Type). Outputs: label id and `float32[1,7]` probabilities.

## Rebuild

```bash
python -m src.train
```

## Results (v1.0.0, 99 rows after cleaning)

| Metric | Value |
| --- | --- |
| 5-fold stratified CV macro F1 | 0.954 +/- 0.064 |
| 5-fold out-of-fold accuracy | 0.980 |
| Holdout (20 rows) macro F1 | 1.000 (very noisy) |
| ONNX vs sklearn label agreement (all rows) | 1.000 |

Small sample: treat these numbers as a rough estimate. See `docs/MODULE_04_PILOT.md`.
