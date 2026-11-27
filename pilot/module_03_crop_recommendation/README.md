# Pilot Module 3: Crop recommendation (ONNX, VRPS-aligned)


Recommends a crop from soil and weather features. Training stays in scikit-learn; the shipped artifact is **ONNX** (same pattern as VRPS `module_01_soil_fertility`).

## Layout

```
data/raw/          # source CSV
data/processed/    # cleaned training table
src/train.py       # rebuild model + metrics
src/infer.py       # ONNX inference API
artifacts/         # ONNX + schema + labels + manifest
results/           # metrics, plots, report
docs/              # design notes
```

## Install

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Train

```powershell
python -m src.train
```

Writes `artifacts/crop_recommendation_v1.onnx`, `results/metrics.json`, confusion matrix, and feature importance. Also refreshes `models/crop_rf.onnx` for compatibility.

## Predict

```powershell
python predict.py --n 90 --p 42 --k 43 --ph 6.5 --temperature 20.9 --humidity 82 --rainfall 203
```

Or the JSON API demo:

```powershell
python -m src.infer
```

## Android / on-device

Copy from `artifacts/`:

- `crop_recommendation_v1.onnx`
- `schema.json`
- `labels.json`

Input: `float32[1,7]` in `schema.json` feature order. Output: class id + probability vector.