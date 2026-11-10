# Module 5 — Crop-yield prediction

Predict recorded yield for a later year from crop, season, state, year, area, and annual rainfall.

## Dataset

Dataset A from the 15 October assignment: crop yields of Indian states and union territories, 1997–2020. The Kaggle page is [akshatgupta7/crop-yield-in-indian-states-dataset](https://www.kaggle.com/datasets/akshatgupta7/crop-yield-in-indian-states-dataset). This folder uses a public mirror of `crop_yield.csv` (19,689 rows). Dataset B, the FAO country table on Zenodo, is a different collection and is not merged in.

| Item | Value |
| --- | --- |
| Rows used | 19,689, after dropping non-positive area if any |
| Train | 16,440 rows, year ≤ 2016 |
| Test | 3,249 rows, 2017–2020 |
| 2020 | 37 rows. The year is incomplete in this file |
| Crops | 55, including Bajra, Small millets, Jowar, Ragi, and Maize |
| States | 30 |

Bajra is the pearl-millet name in this table. Jowar, ragi, and small millets are related grains and are reported as their own slice, not as pearl millet.

## Columns that are not inputs

`Production` is excluded. Yield is approximately production divided by area (median of that ratio is 1.004). Feeding production to the model would hand it the answer.

`Fertilizer` and `Pesticide` are excluded from the main models. Divided by area, they take only a handful of values, and those values follow the year. They behave like a national rate times area. A second random forest puts them back in, on the same split, to measure the difference.

## Run

```bash
pip install -r requirements.txt
python train_baselines.py
```

The split is by year. Rows are not shuffled. A crop that appears in the test years but not in training is still accepted: unseen categories become zeros in the one-hot encoding, and the crop-mean baseline falls back to the global training mean.

Baselines:

- Train-mean: predict the average yield of 1997–2016 for every test row.
- Crop-mean: predict the average training yield of that crop.

## All-crop test scores, 2017–2020

| Model | MAE | RMSE | Median AE | R² |
| --- | --- | --- | --- | --- |
| Train mean | 144.96 | 844.73 | 80.00 | 0.000 |
| Crop mean | 22.84 | 311.30 | **0.447** | 0.864 |
| Linear regression | 56.33 | 309.03 | 22.11 | 0.866 |
| Decision tree | 15.63 | 202.93 | 0.845 | 0.942 |
| Random forest | 12.67 | 182.83 | 0.473 | 0.953 |
| Gradient boosting | **12.39** | **169.23** | 0.840 | **0.960** |
| Linear SVM | 70.66 | 836.55 | 0.378 | 0.019 |

Adding fertilizer and pesticide changes the random-forest MAE from 12.67 to 12.15. That is a small move, and those columns are not crop measurements. They stay out of the model carried forward.

The all-crop R² near 0.96 is mostly crop identity. The random forest assigns importance 0.86 to the coconut indicator. The fifteen largest yields in the file are all coconut, from about 13,000 to 21,105 (`results/largest_recorded_yields.csv`). Bajra's median yield in this file is about 1.0. Those numbers are not the same unit in any practical sense, even though they share a column.

## Same predictions, coconut removed from the test metrics

25 of the 3,249 test rows are coconut. The other 3,224:

| Model | MAE | RMSE | Median AE | R² |
| --- | --- | --- | --- | --- |
| Crop mean | 2.48 | 10.36 | **0.443** | 0.483 |
| Gradient boosting | 2.55 | 10.01 | 0.828 | 0.517 |
| Random forest | **1.65** | **9.03** | 0.467 | **0.607** |

Gradient boosting's win on the full table disappears when coconut is set aside. It is slightly worse than predicting each crop's historical average. Random forest is the model that still improves on that average.

## Bajra only (85 test rows)

| Model | MAE | RMSE | Median AE | R² |
| --- | --- | --- | --- | --- |
| Crop mean | 1.65 | 3.11 | 1.38 | −0.025 |
| Gradient boosting | 1.07 | 2.80 | 0.58 | 0.166 |
| Random forest | **0.81** | **1.04** | 0.64 | **0.885** |

On the wider millet group (bajra, small millets, jowar, ragi, 327 test rows), random-forest median absolute error is 0.37, close to the crop-mean median of 0.40, while R² falls to −1.46. A few rows are dominating R². Median error is the number to read for that slice. The rows were not deleted.

Linear SVM does not fit this target. R² of 0.02 means it is barely better than predicting a constant, because the coconut-scale numbers dominate a linear loss. That is a reason to reject linear SVM here, not a reason to delete coconut before looking.

Plots: `results/residual_scatter.png`, `results/residual_histogram.png`, `results/mae_comparison.png`. The scatter is hard to read until coconut is kept in mind: a few points sit near 20,000 and stretch both axes.

## Findings

1. Most of the impressive R² is "this row is coconut" versus "this row is a grain." Crop-mean alone already reaches R² 0.864.
2. Once coconut is removed from the scoring, random forest is the only candidate that clearly beats the crop average (MAE 1.65 against 2.48).
3. For Bajra, the forest cuts MAE from 1.65 to 0.81 against the crop average. That is the project-relevant slice, and it is 85 test rows, not a field trial.

## Limitations

- Yield units are mixed. Coconut counts are not grain tonnes per hectare. A single MAE across crops is not an agronomic error.
- 2020 has 37 rows. The test window is mostly 2017–2019.
- Area, rainfall, and state are not the project's sensor features. Nothing here has been checked against a millet plot.
- Fertilizer and pesticide in this file are year-level rates times area. Using them would look like a soil feature and would not be one.
- Extreme yields were kept. Removing them would make RMSE look calmer and would hide the unit problem.

## What proceeds

Carry the random forest, not the gradient boosting model, and do not tune either until coconut is scored separately from the grain crops. The forest settings used here are 200 trees, `max_depth=16`, `min_samples_leaf=5`. The next experiment is a Bajra-only model on the same year split, with error reported in the Bajra unit, beside the all-crop model rather than instead of the audit above.
