# Pilot Module 1: Soil fertility (Android-ready)

Classifies soil as **Less Fertile / Fertile / Highly Fertile** from 12 nutrient features.

Inference uses **ONNX Runtime only** (on Android: `onnxruntime-android`). No server, bridge, or tunnel is required for prediction. Preprocessing is inside the ONNX graph.

## Quick start

```bash
cd pilot/module_01_soil_fertility
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.infer
```

## Ship to Android

Copy these files into the app assets:

- `artifacts/soil_fertility_v1.onnx`
- `artifacts/schema.json`
- `artifacts/labels.json`

Input: `float32[1,12]` in `schema.json` feature order. Output: class id + probabilities.

## Rebuild

```bash
python -m src.train
```

## Results (v1.0.0)

| Metric | Value |
| --- | --- |
| Holdout accuracy | 0.983 |
| Holdout macro F1 | 0.979 |
| 5-fold CV macro F1 | 0.956 +/- 0.007 |
| ONNX vs sklearn agreement | 1.000 |

Details: `results/metrics.json`, `docs/MODULE_01_PILOT.md`.
