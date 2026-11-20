# Pilot Module 2: Soil moisture (4-hour ahead, Android-ready)

Predicts Port 1 volumetric water content **4 hours ahead** from recent sensor lags.

Inference uses **ONNX Runtime only** (`onnxruntime-android` on phone). No bridge or tunnel for prediction.

## Honest result (v1.0.0)

**Persistence still wins on MAE.** The shipped Ridge ONNX model is available for the pipeline API, but `schema.json` sets `recommended_policy` to `prefer_persistence_for_4h`. `infer.py` returns both the model value and the persistence value.

| Model | MAE | RMSE | R2 |
| --- | --- | --- | --- |
| Persistence | 0.00214 | 0.00860 | 0.946 |
| Ridge (shipped ONNX) | 0.00249 | 0.00859 | 0.946 |
| Random forest | 0.00290 | 0.00906 | 0.940 |

## Quick start

```bash
cd pilot/module_02_soil_moisture
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.infer
```

## Ship to Android

- `artifacts/soil_moisture_4h_v1.onnx`
- `artifacts/schema.json`
- `artifacts/labels.json`

Input: `float32[1,9]` in schema feature order.

## Data

Zenodo `10.5281/zenodo.20665025`, Port 1, 15-minute series. Chronological 80/20 split. Rainfall/irrigation not joined in v1.
