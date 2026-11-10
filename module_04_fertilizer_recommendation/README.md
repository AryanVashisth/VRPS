# Module 4 — Fertilizer recommendation

Predict `Fertilizer Name` from temperature, humidity, moisture, soil type, crop type, and N, P, K.

## Dataset

Dataset A from the 15 October assignment:

[Fertilizer Prediction.csv on Hugging Face](https://huggingface.co/spaces/SamarthPujari/Crop-and-Fertilizer-Recommendation-System/blob/main/Datasets/Fertilizer%20Prediction.csv)

Saved as `data/raw/fertilizer_prediction.csv`.

Dataset B, the Kaggle playground synthetic table, is not downloaded and is not mixed in. The 100,000-row `Fertilizer Prediction.csv` in the September hardware folder is also a different file. Do not append it to these 99 rows.

| Check | What the file actually contains |
| --- | --- |
| Rows on disk | 198 |
| Unique rows | 99 |
| Why | Every recommendation is stored twice |
| Label error | The grade `10-26-26` was saved as the date `10/26/2026` |
| Classes after the fix | Urea 22, DAP 18, 28-28 17, 14-35-14 14, 20-20 14, 17-17-17 7, 10-26-26 7 |
| Crop value `Millets` | Present as a crop type, not as a field trial of pearl millet |

The source column is spelled `Temparature`. The script renames it to `Temperature` and strips the space in `Humidity`.

## Run

```bash
pip install -r requirements.txt
python train_baselines.py
```

A single 80/20 split would put one or two test rows in `17-17-17` and `10-26-26`. Every model therefore uses the same stratified 5-fold split, seed 42. The confusion matrix is built from out-of-fold predictions, so each of the 99 rows is scored from a model that did not train on it.

The importance plot is a separate fit on all 99 rows. It is there to see which columns the forest uses. It is not the score.

## Results

| Model | Accuracy mean | Accuracy std | Macro F1 mean | Macro F1 std |
| --- | --- | --- | --- | --- |
| Random forest | 0.990 | 0.020 | **0.967** | 0.065 |
| Gradient boosting | 0.990 | 0.020 | **0.967** | 0.065 |
| RBF SVM | 0.938 | 0.052 | 0.885 | 0.066 |
| Decision tree | 0.899 | 0.054 | 0.791 | 0.083 |
| Majority class | 0.222 | 0.023 | 0.052 | 0.004 |

Out-of-fold, the random forest is perfect on six of the seven grades. The miss is `17-17-17`: precision 1.00, recall 0.86, F1 0.92, on 7 rows. That one miss is why the fold-to-fold macro F1 moves by about 0.07. An accuracy of 0.99 and a standard deviation of 0.07 are both true.

Impurity importance: phosphorus 0.28, nitrogen 0.23, potassium 0.21. Soil type and crop type, including Millets, are far smaller. Open `results/random_forest_importance.png` and `results/confusion_matrix_best.png`.

## Findings

1. The raw file is 99 recommendations duplicated, plus one grade rewritten as a date. Both have to be fixed before a score means anything.
2. Random forest and gradient boosting tie. The forest is the one to keep, because it accepts `class_weight="balanced"` and the importance plot is direct.
3. N, P, and K dominate. On this table the fertilizer name is mostly a nutrient-grade lookup, which is why a small model looks almost perfect.

## Limitations

- 99 rows, and two grades have 7 rows. The standard deviation across folds is the honest uncertainty.
- These labels are fertilizer product names from a public demo table. They are not prescriptions checked against the project soils.
- A near-perfect score on a lookup table is not evidence the same rules will hold for sensor N, P, and K.
- Gradient boosting has no class weight in scikit-learn, so the tie with the forest should not be read as a careful imbalance treatment.
- CatBoost was not added. The scikit-learn gradient boosting model is the boosting candidate, and it did not beat the forest.

## What proceeds

Keep the random forest as Model v1 for this module. Do not grid-search it. The next experiment is a larger set of recommendations whose labels were not generated from the same NPK rules, or a check of whether the seven `17-17-17` rows are the only hard cases for a reason visible in N, P, and K.

Tree depth is capped (`max_depth=4`, `min_samples_leaf=3`) because a deep tree on 99 rows will memorize grades. The forest uses 200 trees and `min_samples_leaf=2`.
