# Module 1 — Soil / nutrient analysis

Predict the fertility class that is already in the file. The labels are not created by this script.

| Code | Class |
| --- | --- |
| 0 | Less Fertile |
| 1 | Fertile |
| 2 | Highly Fertile |

That mapping is the one published with the dataset author's README. Inputs are N, P, K, pH, EC, organic carbon, S, Zn, Fe, Cu, Mn, and B.

## Dataset

| Field | Value |
| --- | --- |
| Assigned pointer | [Soil fertility notebook](https://www.kaggle.com/code/nouraalgohary/soil-fertility-prediction/notebook) |
| File used | Original table linked from that notebook's README, saved as `data/raw/soil_fertility.csv` |
| Not used | The author's modified table, which adds rows for the rare class |
| Rows in the file | 1,288 |
| Unique rows | 880 |
| Licence | Not stated on the GitHub copy. Confirm before anything leaves the project. |
| Geography | Not an Indian millet field trial |

The published file looks balanced (401 / 440 / 447). After exact copies are removed it is not:

| Class | Rows in the file | Unique rows |
| --- | --- | --- |
| Less Fertile | 401 | 401 |
| Fertile | 440 | 440 |
| Highly Fertile | 447 | 39 |

All 408 extra rows are copies of Highly Fertile samples. Training on the file as published puts copies on both sides of a random split.

## Run

```bash
pip install -r requirements.txt
python train_baselines.py
```

`train_baselines.py` writes `results/`. It fits five models twice: once on the unique rows, and once on the published file so the leak is visible. The second run is labeled `invalid_with_duplicate_rows`. Do not quote it as the module score.

Split for the real comparison: stratified 80/20, seed 42, after duplicates are removed. 704 train rows, 176 test rows. Scaling and median imputation are inside the pipeline.

## Results on the 880 unique rows

| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
| --- | --- | --- | --- | --- |
| Majority class | 0.500 | 0.167 | 0.333 | 0.222 |
| Decision tree | 0.932 | 0.835 | 0.878 | 0.853 |
| Logistic regression | 0.813 | 0.663 | 0.720 | 0.666 |
| Random forest | 0.932 | 0.853 | 0.877 | **0.864** |
| RBF SVM | 0.835 | 0.693 | 0.773 | 0.708 |

On the duplicated file, the same random forest reaches macro F1 0.961, and 90 test rows are exact copies of training rows. That higher number is the leak, not a better model.

Random-forest class scores on the clean test set:

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| Less Fertile | 0.94 | 0.95 | 0.94 | 80 |
| Fertile | 0.95 | 0.93 | 0.94 | 88 |
| Highly Fertile | 0.67 | 0.75 | 0.71 | 8 |

Open `results/confusion_matrix_best.png` and `results/random_forest_importance.png`.

The forest's impurity importance is concentrated on N (0.45) and P (0.26). K is near the bottom (0.02), close to pH and the micronutrients. Do not read that as "potassium does not matter in soil." It means, in this table, potassium did not separate the three classes once N and P were known.

## Findings

1. The file's class balance is an artifact of copied Highly Fertile rows. Macro F1 is the score to report, because accuracy treats the 8 rare test rows as optional.
2. Random forest is the best of the five models on the clean split, and the depth-limited tree is close (0.853 vs 0.864). The algorithm is not the bottleneck.
3. Logistic regression and the RBF SVM lose most of their macro F1 on Highly Fertile. A single overall accuracy hides that.

## Limitations

- Highly Fertile has 39 unique rows and 8 test rows. One or two mistakes move that class's F1 a lot.
- Nothing here was measured on the project sensors, and the table is not a millet trial.
- Impurity importance is not a causal ranking of nutrients.
- The modified dataset that oversamples the rare class was left unused on purpose. Adding copies would repeat the leak.

## What proceeds

Carry the random forest forward only as a cleaned baseline. The next experiment is more unique Highly Fertile samples, or a review of whether those 39 rows are themselves near-copies. Tuning depth and tree count on 39 rows will not fix the class.

Parameters worth knowing: tree `max_depth=5`, `min_samples_leaf=5`; forest 200 trees, `min_samples_leaf=5`, `class_weight="balanced"`.
