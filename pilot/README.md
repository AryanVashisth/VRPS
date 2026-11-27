# VRPS pilot builds

Production-leaning modules that can ship into an Android app and later plug into one end-to-end advisory pipeline.

| Module | Original map | Status | Runtime on phone |
| --- | --- | --- | --- |
| `module_01_soil_fertility` | #1 Soil | v1.0.0 ready | ONNX Runtime Mobile |
| `module_02_soil_moisture` | #2 Moisture | v1.0.0 ready (persistence preferred) | ONNX Runtime Mobile |
| `module_03_crop_recommendation` | #6 Crop | v1.1.0 ready | ONNX Runtime Mobile |
| `module_04_fertilizer_recommendation` | #4 Fertilizer | v1.0.0 ready (99 rows; trust CV F1) | ONNX Runtime Mobile |
| `module_05_crop_yield` | #5 Yield | v1.0.0 ready (report Bajra slice) | ONNX Runtime Mobile |
| `module_06_pearl_millet_disease` | #3 Disease | v1.0.0 ready (partial classes; colour features) | ONNX after feature extract |

Convention for every pilot module:

- Train with scikit-learn (or convertible tooling)
- Export **ONNX** with preprocessing in-graph when practical
- Ship `schema.json` + labels/maps + model
- Infer with onnxruntime only (no network for prediction)

## Pilot Console

Local dummy dashboard to smoke-test all six modules:

```bash
cd dashboard
../.venv/bin/uvicorn app:app --port 8765
```

Open http://127.0.0.1:8765

## Timeline note

Meets 30 to 33 cover the sensor path into Module 1. Meets 34 to 37 freeze Modules 4 to 6 and the console, then return to Month 6 integration prep. Do not treat the console as field accuracy.
