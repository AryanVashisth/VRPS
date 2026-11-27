# Pilot Module 3: Crop recommendation (Android-ready)

Recommends a crop label from N, P, K, temperature, humidity, pH, rainfall.

**Honesty:** public augmented table, 22 crops, **no millet class**. High Macro F1 describes this table, not a millet field.

## Quick start
```bash
pip install -r requirements.txt
python -m src.infer
```

## Ship
`artifacts/crop_recommendation_v1.onnx` + `schema.json` + `labels.json` (onnxruntime-android).

## Sensor note
N/P/K/pH can come from soil tests/sensors. Temperature/humidity from environment sensors. Rainfall needs a rain gauge or weather source, not a soil probe alone.
