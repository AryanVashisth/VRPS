# Module 1 pilot: design and deployment notes

Date: 14 November 2026  
Version: 1.0.0  
Path: `pilot/module_01_soil_fertility/`

## Goal

Deliver a soil-fertility classifier that can run on an Android phone offline, and that fits the first slot of the end-to-end VRPS pipeline (sensors/forms -> model -> farmer-readable result).

## Why ONNX (not sklearn pickle, not Torch)

| Option | On Android without a bridge? |
| --- | --- |
| scikit-learn `.joblib` | No native runtime |
| PyTorch | Heavy; needs conversion |
| TensorFlow Lite | Possible, but RF export is awkward |
| **ONNX + onnxruntime-android** | Yes. Small CPU runtime, same file on phone and laptop |

Training stays in scikit-learn. The shipped artifact is ONNX. The Python `src/infer.py` API mirrors what the Android app should call.

## Data (India-first)

Combined processed file: `data/processed/soil_fertility_india_v1.csv` (5,921 rows after cleaning).

1. **India Soil Health Card, Ahmedpur, Latur, Maharashtra** (project-mitti public export)  
   Parsed N, P, K, pH, EC, OC, S, Zn, Fe, Cu, Mn, B. Fertility labels from SHC L/M/H status on N, P, K, OC.
2. **Public labeled soil fertility table** (deduplicated; original Module 1 labels 0/1/2).
3. **India crop-recommendation NPK rows** (600 stratified) with rule labels from N/P/K bands. Weak prior only; micronutrients missing and imputed.

### Important honesty note

Most India SHC labels are **derived from nutrient status codes**, not a separate lab fertility grade. The model is strong at recovering that structure. Pilot use is valid for nutrient-status style fertility. Campus sensor validation is still required before claiming field accuracy.

## Features (fixed order)

`N, P, K, ph, ec, oc, S, zn, fe, cu, Mn, B`

Do not reorder. Android and Python must use `schema.json`.

## Model

Pipeline inside ONNX:

1. Median imputer  
2. Standard scaler  
3. RandomForestClassifier (200 trees, max_depth=12, min_samples_leaf=3, class_weight=balanced, seed=42)

## Metrics

Holdout (20%, stratified, seed 42):

- Accuracy 0.983  
- Macro F1 0.979  
- Highly Fertile support on test: 50 rows (much stronger than the old 8-row baseline slice)

5-fold CV macro F1: 0.956 +/- 0.007  
ONNX label agreement with sklearn on the holdout: 100%

## End-to-end pipeline slot

```
Sensor / form input (12 floats)
        |
        v
schema.json feature order
        |
        v
soil_fertility_v1.onnx  (on-device)
        |
        v
{fertility_id, fertility, confidence, probabilities}
        |
        v
Later: decision engine (other modules attach here)
```

## Android integration sketch

1. Add `onnxruntime-android` dependency.  
2. Load `soil_fertility_v1.onnx` from assets.  
3. Build `float[1][12]` in schema order (use NaN for missing; imputer is in-graph).  
4. Read label output and probability output.  
5. Map id through `labels.json`.

No HTTP call is required for inference.

## Limits / next hardening

- Add more Indian districts (full SHC extracts when open download is available).  
- Calibrate units from campus sensors against this feature schema.  
- Keep Highly Fertile monitoring: still the smallest class.  
- Do not reintroduce server-side inference for the pilot phone path.
