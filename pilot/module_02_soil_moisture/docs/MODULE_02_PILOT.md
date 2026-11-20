# Module 2 pilot: design notes

Version: 1.0.0  
Path: `pilot/module_02_soil_moisture/`

## Goal

Second pipeline slot: forecast soil water content 4 hours ahead, on-device via ONNX, same contract style as Module 1.

## Why persistence is recommended

On the chronological holdout, no trained model beat persistence MAE. That matches the earlier baseline finding. The app should prefer `lag_0` (no change) for the 4-hour call unless a later version with rainfall beats it.

## Features (fixed order)

`lag_0, lag_1, lag_4, lag_16, lag_96, mean_last_4h, std_last_4h, delta_1, delta_4`

## Artifacts

- `soil_moisture_4h_v1.onnx` (Ridge pipeline with imputer + scaler)
- ONNX vs sklearn agreement: 100% (within 1e-5)

## Limits

- Italy olive-field sensors, not millet project hardware
- No irrigate / do-not-irrigate label
- Next upgrade: add rainfall/irrigation from the same Zenodo zip and re-compare to persistence
