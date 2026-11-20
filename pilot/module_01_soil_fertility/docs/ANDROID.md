# Android on-device inference

Dependency: `com.microsoft.onnxruntime:onnxruntime-android`

Assets to bundle:

- `soil_fertility_v1.onnx`
- `schema.json` (feature order)
- `labels.json`

Input shape: `float32[1][12]` in schema order. Missing values: `Float.NaN` (imputer is inside ONNX).

Outputs: class label + probability vector. Map label id with `labels.json`.

No HTTP, tunnel, or desktop bridge is required for prediction.
