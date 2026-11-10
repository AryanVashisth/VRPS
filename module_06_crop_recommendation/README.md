# Module 6 — Crop recommendation

Classify the crop label from N, P, K, temperature, humidity, pH, and rainfall.

## Dataset

Assigned dataset: [Kaggle crop recommendation](https://www.kaggle.com/datasets/siddharthss/crop-recommendation-dataset).

The file in `data/raw/crop_recommendation.csv` is a public copy of that table: 2,200 rows, 22 crops, 100 rows each, no duplicate rows. The dataset description says it was built by augmenting Indian rainfall, climate, and fertilizer data. It is not a set of independent field trials.

There is no pearl millet, bajra, jowar, or ragi class. Maize is the closest grain in the file, and it is not a substitute.

The 20,000-row sensor crop table and the Nashik spreadsheet in the September meeting folder are different datasets. They are not used here.

## Run

```bash
pip install -r requirements.txt
python train_baselines.py
```

Split: stratified 80/20, seed 42. 1,760 train rows, 440 test rows. The same rows are used for every model. Scaling is inside the pipeline.

The script then refits nothing for the pH check. It takes the already trained random forest, holds each probe crop at its training median, and changes only pH from 4.5 to 8.5 in steps of 0.5. Probe crops are rice, maize, cotton, and chickpea.

## Results

| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
| --- | --- | --- | --- | --- |
| Majority class | 0.045 | 0.002 | 0.045 | 0.004 |
| Decision tree | 0.870 | 0.857 | 0.870 | 0.852 |
| Logistic regression | 0.973 | 0.974 | 0.973 | 0.972 |
| RBF SVM | 0.984 | 0.986 | 0.984 | 0.984 |
| Random forest | 0.993 | 0.994 | 0.993 | **0.993** |

The misses are single test rows. Recall is 0.95 for blackgram, lentil, and rice (one miss out of 20). Precision is 0.95 for jute, maize, and mothbeans. Every other class is perfect on this split. The confusion matrix is `results/confusion_matrix_best.png`.

Random-forest importance, largest first: rainfall 0.222, humidity 0.216, K 0.180, P 0.153, N 0.105, temperature 0.071, pH 0.052. pH is the smallest of the seven inputs.

pH sweep: for all four median profiles, and all nine pH values, the predicted crop stayed the profile's crop. Flip count is 0. The table is `results/ph_sensitivity.csv`.

## Findings

1. The classes in this file are easy to separate. A majority guess scores about 4.5% accuracy. The forest scores 99.3% macro F1. That gap describes the table, not a farmer's field.
2. pH is the weakest input in the forest. Moving pH across 4.5–8.5 on a typical rice, maize, cotton, or chickpea profile does not change the recommended crop.
3. Rainfall and humidity carry more of the decision than N, P, K, or pH. A soil-only reading, without those climate columns, is a different model from the one scored here.

## Limitations

- No millet class. This module cannot recommend pearl millet, and a high score must not be quoted as millet performance.
- The source description says the rows were augmented. Near-perfect accuracy is what a clean, generated boundary looks like.
- The pH sweep uses median profiles, which sit in the middle of each crop's cloud. It does not prove pH is irrelevant near a boundary, or on a real sensor reading outside these ranges.
- The earlier project note, that lowering pH produced a poor recommendation, is not reproduced by this forest on these four profiles. That note may have come from another script, another profile, or inputs this table never contained. This result does not close that issue by itself.
- Logistic regression reaches 0.972, which fits a table whose classes are almost linearly separable. That is another sign the file is easy, not a reason to deploy the coefficients.

## What proceeds

The random forest is the strongest candidate and the one with a readable importance plot. Do not spend the next cycle tuning it toward 0.995. The useful next checks are a profile near a class boundary, and a dataset that actually contains the millet crops the project is about.

The depth-limited tree (macro F1 0.852) is in the comparison so the forest's score has something simple beside it. Parameters: forest of 200 trees, `min_samples_leaf=2`; tree `max_depth=8`, `min_samples_leaf=5`.
