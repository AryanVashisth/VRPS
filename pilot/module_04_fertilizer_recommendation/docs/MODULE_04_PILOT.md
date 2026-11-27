# Module 4 pilot: fertilizer recommendation

Version 1.0.0. Path: `pilot/module_04_fertilizer_recommendation/`

## Data and cleaning

Source: `data/raw/fertilizer_prediction.csv` (198 rows). Cleaning in `src/train.py`:

- Exact duplicate rows dropped: 198 -> **99 rows**.
- Label `10/26/2026` (a date read from the grade) mapped back to `10-26-26`.
- `Temparature` renamed `Temperature`.

Classes: Urea 22, DAP 18, 28-28 17, 14-35-14 14, 20-20 14, 10-26-26 7, 17-17-17 7.

## Model

scikit-learn pipeline: median imputer on six numeric columns, one-hot on Soil Type and Crop Type, then `RandomForestClassifier` (200 trees, `min_samples_leaf=2`, `class_weight=balanced`, seed 42). This was the best model in the development baselines. The ONNX conversion with string inputs and one-hot worked, so category maps are not needed. The shipped model is fit on all 99 rows (479 KB).

## Metrics

- 5-fold `StratifiedKFold` macro F1: 0.954 +/- 0.064; out-of-fold accuracy 0.980. This is the number to quote.
- Single stratified 80/20 holdout (20 test rows): macro F1 1.000. With 1-2 rows in the small classes this says little.
- ONNX vs sklearn: labels agree on all 99 rows, max probability difference about 2e-7. This check is on training rows, so it tests conversion, not generalization.

## Honest limits

- 99 rows is small. Several classes have 7 rows, and a miss moves per-class F1 a lot.
- The data looks synthetic or tightly rule-generated. High scores may reflect clean structure, not field accuracy.
- The label is a fertilizer product, not an agronomic dose. Not validated against local soil-test recommendations.
- Categories outside the training lists (see `allowed` in `schema.json`) are ignored by the one-hot, so output is less reliable for them.

## Android

Add `onnxruntime-android`, load `model.onnx` from assets, feed the eight inputs listed in `schema.json`, map the label id through `labels.json`. No network call is involved.
