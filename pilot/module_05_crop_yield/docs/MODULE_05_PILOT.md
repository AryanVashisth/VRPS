# Module 5 pilot: crop yield

Version 1.0.0. Path: `pilot/module_05_crop_yield/`

## Data and split

Source: `data/raw/crop_yield.csv` (19,689 rows, all used after checks; whitespace in Crop/Season/State is stripped).

Chronological split: train `Crop_Year <= 2016` (16,440 rows), test 2017-2020 (3,249 rows; 2020 has only 37).

## Features

Crop_Year, Area, Annual_Rainfall, Crop, Season, State.

Left out: `Production` (it reconstructs Yield), `Fertilizer` and `Pesticide` (year-level rates times area, not soil or farm measurements).

Categoricals are integer codes from maps fit on the train split only, stored in `schema.json`. Unknown values get `-1`. This keeps the ONNX file at 9.6 MB instead of a wide one-hot tree model. All test rows had categories seen in training, so the unknown path is untested on real data.

## Model

`RandomForestRegressor` (100 trees, `max_depth=16`, `min_samples_leaf=5`, seed 42). Shipped model is trained on the train split only, so reported metrics match the file.

## Metrics (test 2017-2020)

| Slice | n | RF MAE | RF median AE | Crop-mean MAE |
| --- | --- | --- | --- | --- |
| All rows | 3,249 | 11.74 | 0.29 | 22.84 |
| Excluding coconut | 3,224 | 1.46 | 0.29 | 2.48 |
| Bajra (pearl millet) | 85 | 0.67 | 0.42 | 1.65 |

ONNX vs sklearn on the test set: max absolute difference 0.0013 (float32 rounding on large coconut values), relative about 2e-7.

## Honest limits

- Coconut yields are orders of magnitude above grains and account for about 88% of total absolute error. The all-rows MAE mostly measures coconut. Use the non-coconut and Bajra rows for grain use.
- Crop identity explains most of the model (importance about 0.85), then state (0.11). Year, area and rainfall add little, so this is closer to a crop/state average than a weather-responsive forecast.
- Persistence (last year's yield for the same crop/state/season) is not evaluated: the table is not a clean panel.
- The Bajra slice is 85 rows. Do not read fine differences into it.
- Rainfall is a state-year figure, not field-level. Not validated on campus or farm records.

## Android

Add `onnxruntime-android`, load `model.onnx`, apply the code maps from `schema.json`, feed `float32[1,6]`, read one float. No network call is involved.
