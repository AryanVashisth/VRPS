# Module 6 pilot: pearl millet leaf disease

Date: 27 November 2026
Version: 1.0.0
Module id: `module_06_pearl_millet_disease`
Path: `pilot/module_06_pearl_millet_disease/`
Baseline: `development/module_03_pearl_millet_disease/`

## Limits

1. **Partial classes.** Three classes: Exserohilum rostratum (leaf spot), healthy, Sclerospora graminicola (downy mildew). The smut and Pyricularia archives from the source data were not downloaded, so the model will label a smut or blast leaf as one of the three classes above. Do not show it to farmers as a full pearl millet disease checker.
2. **Tiny, imbalanced data.** 80 images: Exserohilum 9, Healthy 41, Downy mildew 30. The holdout has 16 images (2 Exserohilum).
3. **Optimistic split.** The Zenodo folders carry no plant or leaf IDs. A random split can put photos of the same leaf in train and test. The 1.000 holdout score reflects that and the tiny test set. Treat the 5-fold CV macro F1 of 0.884 +/- 0.071 as an upper-leaning estimate. Field accuracy is unknown.
4. **Colour features, not a CNN.** Per channel: mean, std, 8-bin histogram on a 64x64 resize. Background, lighting, and camera differences can shift these numbers. Lesion shape and texture are ignored.
5. **Demo images come from the training set.** They are a smoke test of the pipeline, not a validation.

## Features (fixed order, 30 floats)

`mean_r, mean_g, mean_b, std_r, std_g, std_b, hist_r_0..7, hist_g_0..7, hist_b_0..7`

Computed by `src/features.py` (`colour_features`, `feature_names`): open as RGB, resize to 64x64 (PIL default resampling), divide by 255, per-channel mean and population std, then `np.histogram` with 8 bins over [0, 1] normalised to sum 1. This matches the development `train_baselines.py`.

## Model

Pipeline exported to ONNX: median imputer, standard scaler, RandomForestClassifier (200 trees, min_samples_leaf=3, class_weight=balanced, seed 42). ONNX input: `float32[N,30]`. Outputs: label and probabilities (zipmap off).

## Labels

| id | Name | Source folder |
| --- | --- | --- |
| 0 | Exserohilum leaf spot | `exserohilum_rostratum` |
| 1 | Healthy | `healthy` |
| 2 | Downy mildew | `sclerospora_graminicola` |

## Artifacts

`artifacts/`: `disease_v1.onnx`, `schema.json`, `labels.json`, `manifest.json` (SHA-256 hashes).
`results/`: `metrics.json`, `confusion_matrix.png`, `classification_report.txt`.
`data/processed/pearl_millet_colour_features_v1.csv`: one row per image with the 30 features.
`data/demo_images/`: 6 sample leaves (2 per class) for a later dashboard.

## Android

ONNX Runtime on Android (`onnxruntime-android`) can run `disease_v1.onnx`, but **image to feature extraction is not in the graph**. Two options:

1. **Reimplement in the app.** Resize to 64x64 RGB, scale to 0-1, compute the 30 features in the order in `schema.json`, feed `float32[1,30]`. Resize interpolation and rounding differ between PIL and Android bitmaps, so check parity against `data/processed/*.csv` before trusting results.
2. **Call shared feature code** (for example a small server or Python/Chaquopy layer running `src/features.py`), then run ONNX locally.

Option 2 gives exact parity with training. Option 1 keeps it offline.

## Next steps

- Download the smut and Pyricularia archives and retrain with 5 classes.
- Collect plant-level IDs and split by plant.
- Compare a small CNN (for example MobileNet) against this colour baseline.
