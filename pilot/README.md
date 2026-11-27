# VRPS pilot builds

Production-leaning modules that can ship into an Android app and later plug into one end-to-end advisory pipeline.

| Module | Status | Runtime on phone |
| --- | --- | --- |
| `module_01_soil_fertility` | v1.0.0 pilot ready | ONNX Runtime Mobile |
| `module_02_soil_moisture` | v1.0.0 pilot ready (persistence preferred) | ONNX Runtime Mobile |
| `module_03_crop_recommendation` | v1.1.0 pilot ready (presented pack) | ONNX Runtime Mobile |

Convention for every pilot module:

- Train with scikit-learn (or convertible tooling)
- Export **ONNX** with preprocessing in-graph
- Ship `schema.json` + `labels.json` + model
- Infer with onnxruntime only (no network for prediction)
