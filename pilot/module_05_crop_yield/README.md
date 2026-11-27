# Pilot Module 5: Crop yield

Predicts recorded crop yield (Production / Area in the source table) from year, area, rainfall, crop, season and state.

Inference uses **ONNX Runtime only** (on Android: `onnxruntime-android`). No network is needed. Category-to-integer maps live in `artifacts/schema.json` and are applied by the caller before the ONNX call.

## Quick start

```bash
cd pilot/module_05_crop_yield
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.infer
```

```python
from src.infer import CropYieldModel
CropYieldModel().predict_one({
    "Crop_Year": 2018, "Area": 5000, "Annual_Rainfall": 900.0,
    "Crop": "Bajra", "Season": "Kharif", "State": "Maharashtra",
})
# {yield_pred, model_version}
```

## Ship to Android

Copy `artifacts/model.onnx` and `schema.json`. Build `float32[1,6]` in `feature_order`: Crop_Year, Area, Annual_Rainfall, Crop_code, Season_code, State_code. Trim names, look up codes in `categorical_encoding.maps`, use `-1` if unknown.

## Rebuild

```bash
python -m src.train
```

## Results (v1.0.0, train <= 2016, test 2017-2020)

| Slice | RF MAE | Crop-mean baseline MAE |
| --- | --- | --- |
| All test rows (3,249) | 11.74 | 22.84 |
| Excluding coconut | 1.46 | 2.48 |
| Bajra only (85) | 0.67 | 1.65 |

Coconut accounts for about 88% of total absolute error. See `docs/MODULE_05_PILOT.md`.
