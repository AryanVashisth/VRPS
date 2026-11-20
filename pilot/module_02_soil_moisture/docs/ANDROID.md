# Android on-device inference (Module 2)

Dependency: `com.microsoft.onnxruntime:onnxruntime-android`

Bundle:

- `soil_moisture_4h_v1.onnx`
- `schema.json`
- `labels.json`

Build `float[1][9]` in schema order. Read float output.

If `recommended_policy` is `prefer_persistence_for_4h`, display or act on `lag_0` unless you intentionally show the model value for debugging.
