# Module 2 — Irrigation / soil-moisture forecast

Forecast soil water content. This dataset does not contain an irrigate / do-not-irrigate label, and it is not a millet field.

The series is the Zenodo record [10.5281/zenodo.20665025](https://zenodo.org/records/20665025), licence CC BY 4.0. Measurements are every 15 minutes from an experimental field in Gallipoli, Italy, June 2021 through February 2023. The clean CSV in this folder has two sensors:

| Column | Depth, from the Zenodo description |
| --- | --- |
| `port1_water_content` | 30 cm (odd port) |
| `port2_water_content` | 60 cm (even port) |

The other four ports, the rainfall workbook, and the irrigation workbook are in the same zip. They are not inputs in this baseline. `download_data.py` can fetch the zip.

Observed port 1 values sit between 0.142 and 0.339. That is a fraction of water content, not a percentage and not a yes/no decision.

## Question this script answers

If we already know the recent water content, do the candidate models forecast the future better than the rule "it will stay where it is"?

| Horizon | Steps of 15 min | Role |
| --- | --- | --- |
| 15 minutes | 1 | Shows how slowly the sensor moves |
| 4 hours | 16 | Headline comparison |
| 24 hours | 96 | Longest horizon in this script |

The split is the calendar, not a shuffle. Training timestamps are before 2022-10-30 08:27. Later timestamps are the test period. For the 4-hour port 1 task that leaves 44,280 training rows and 1,534 test rows. A row is kept only when the lags, the 24-hour mean, and the future target are all present, so gaps are not filled with invented readings.

## Run

```bash
pip install -r requirements.txt
python train_baselines.py
```

Plots: `results/mae_comparison_4h.png`, `results/forecast_last_14_test_days.png`, `results/residual_histogram.png`.

## Port 1 results

MAE and RMSE are in water-content units. A difference of 0.001 is one-tenth of one percent volumetric water content.

### 15 minutes

| Model | MAE | RMSE | R² |
| --- | --- | --- | --- |
| Persistence | 0.000076 | 0.000312 | 0.999 |
| Yesterday | 0.003251 | 0.007201 | 0.612 |
| Linear regression | 0.000124 | 0.000336 | 0.999 |
| Random forest | 0.000201 | 0.000336 | 0.999 |
| Linear SVM | 0.000444 | 0.000574 | 0.998 |

### 4 hours (headline)

| Model | MAE | RMSE | R² |
| --- | --- | --- | --- |
| Persistence | **0.000726** | 0.002543 | 0.951 |
| Yesterday | 0.003373 | 0.007400 | 0.586 |
| Linear regression | 0.001232 | 0.002397 | 0.957 |
| Random forest | 0.002209 | 0.004133 | 0.871 |
| Linear SVM | 0.000907 | 0.002382 | 0.957 |

### 24 hours

| Model | MAE | RMSE | R² |
| --- | --- | --- | --- |
| Persistence | 0.003672 | 0.007942 | 0.495 |
| Yesterday | 0.003672 | 0.007942 | 0.495 |
| Linear regression | 0.005629 | 0.007859 | 0.506 |
| Random forest | 0.010022 | 0.012534 | −0.258 |
| Linear SVM | 0.003257 | 0.008109 | 0.474 |

Yesterday matches persistence at 24 hours because "same clock time yesterday" is the current reading when the horizon is one day. That is expected, not a bug.

Port 2 at 4 hours is the same story, more strongly. Persistence MAE is 0.000200 and R² is 0.9996. Every fitted model is worse. See `results/port2_4h_comparison.csv`.

The random forest puts 97.5% of its impurity importance on the current reading (`lag_0`). It has little else to use, and the extra features make the 4-hour and 24-hour forecasts worse than doing nothing.

Linear SVM printed a convergence warning at `max_iter=3000`. Its small MAE edge at 24 hours is not a result to keep. The fit did not finish, and R² did not improve on persistence.

ARIMA is not in the comparison. Refitting ARIMA at every test timestamp on two years of 15-minute data is a separate experiment. The linear model already is an autoregression on the same lags.

## Findings

1. At 15 minutes and at 4 hours, "the water content will not change" is the model to beat. The fitted models do not beat it on MAE.
2. At 24 hours, persistence still explains about half the variance (R² 0.495). The random forest is worse than predicting the training pattern of "no change."
3. The current reading dominates the forest. Rainfall and irrigation, which are in the Zenodo zip and not in this feature set, are the next information worth adding. Another tree model is not.

## Limitations

- This is an olive field in Italy, not a millet field in the project area.
- A strong forecast of water content is not an irrigation decision. The dataset has no such label.
- The test period is the last 20% of the calendar, and a row is dropped when the previous day is incomplete. The 1,534 test rows are the complete windows, not every timestamp after the cutoff.
- Port 1 has 9,866 missing points on the 15-minute grid, in 197 stretches. The longest hole is 5,745 minutes, just under four days. Those stretches are skipped rather than interpolated.
- No model here should be connected to the project sensors. Those readings are not calibrated against this unit.

## What proceeds

Do not tune the forest or the SVM against this score. The next experiment is a 4-hour forecast that adds the rainfall and irrigation series from the same Zenodo record, still compared with persistence, still on a chronological split.
