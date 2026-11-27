# Pilot Module 6: Pearl millet leaf disease (colour features, partial classes)

Classifies a pearl millet leaf photo as **Exserohilum leaf spot / Healthy / Downy mildew** (*Sclerospora graminicola*) using 30 colour features and a Random Forest, exported to ONNX.

**Read this first**

- Only 3 of the 5 planned classes. Smut and Pyricularia (blast) are **not** covered.
- 80 images total (9 / 41 / 30). The test set is 16 images, 2 of them Exserohilum.
- Random image split, no plant or leaf IDs. Near-duplicate photos can sit on both sides, so scores are optimistic.
- This is a colour-summary model, **not** a CNN. It cannot see lesion shape or texture.
- The ONNX input is the 30 features, not an image.

## Quick start

```bash
cd pilot/module_06_pearl_millet_disease
python3 -m venv .venv && source .venv/bin/activate   # do not commit .venv
pip install -r requirements.txt
python -m src.infer                                   # runs on a demo image
python -m src.infer data/demo_images/healthy__PESGL_1.jpg
```

```python
from src.infer import predict_from_image, predict_from_features
predict_from_image("leaf.jpg")
predict_from_features([...30 floats in schema.json order...])  # or a dict by name
```

Output: `disease_id, disease, confidence, probabilities, model_version`.

## Rebuild

```bash
python -m src.train
```

Reads images from the development module (`DEV_IMAGES` env var overrides the path). If that folder is missing it uses `data/processed/pearl_millet_colour_features_v1.csv`.

## Results (v1.0.0, 20% stratified holdout, seed 42)

| Metric | Value |
| --- | --- |
| Holdout accuracy | 1.000 (16 images) |
| Holdout macro F1 | 1.000 |
| 5-fold CV macro F1 | 0.884 +/- 0.071 |
| ONNX vs sklearn agreement | 1.000 |

The holdout number is too small and too leaky to trust. The CV figure is a better guide, and it is also optimistic for the same reason.

Details: `results/`, `docs/MODULE_06_PILOT.md`.
