# Crop recommendation — design notes

Version: 1.1.0  
Aligned with: [VRPS `module_01_soil_fertility`](https://github.com/AryanVashisth/VRPS/tree/pilot/pilot/module_01_soil_fertility)

## Goal

Recommend a crop from soil and weather features, with an **ONNX** artifact that can run offline (including Android via `onnxruntime-android`).

## Features (fixed order)

`N, P, K, pH, Temperature, Humidity, Rainfall`

Do not reorder. Python and Android must follow `artifacts/schema.json`.

## Pipeline (inside ONNX)

1. Median imputer  
2. Optional StandardScaler (only if selected)  
3. RandomForestClassifier (`class_weight=balanced_subsample`, seed=42). Variant chosen by holdout macro F1 — see `artifacts/schema.json`.

## Data

`data/raw/india_crop_recommendation_npk.csv` → cleaned `data/processed/crop_recommendation_v1.csv`  
(~2,200 rows, 22 crop classes after light bound filters / dedupe).

## Rebuild

```powershell
python -m src.train
python -m src.infer
```

## Ship to Android

Copy `artifacts/crop_recommendation_v1.onnx`, `schema.json`, `labels.json`.
