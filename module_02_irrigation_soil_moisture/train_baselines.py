"""Module 2 — forecast soil water content from earlier sensor readings.

The assigned Zenodo record is a 15-minute series from an olive field in
Gallipoli, Italy. It does not contain a millet irrigation yes/no label.
The target is the sensor's own future water content.

Port 1 is the series used for the model comparison. The file's second column
is port 2, at the other depth, and the same 4-hour check is repeated there.

Horizons
    1 step  = 15 minutes
    16 steps = 4 hours     (this is the headline horizon)
    96 steps = 24 hours

The split is chronological: the last 20% of the calendar span is the test
period. Nothing is shuffled.

Persistence predicts "the water content will stay where it is now".
Yesterday predicts "it will match the value at this clock time yesterday".
ARIMA is not fit. A fresh ARIMA at every test timestamp, on two years of
15-minute data, is a different experiment, and the linear lag model already
covers a plain autoregression.

Rainfall and irrigation files ship in the same Zenodo zip. They are not
joined here, because this first baseline answers a narrower question:
how much do recent water-content readings beat persistence?
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR

SEED = 42
HEADLINE_HORIZON = 16
HORIZONS = {
    "15_min": 1,
    "4_hour": 16,
    "24_hour": 96,
}
LAGS = [0, 1, 4, 16, 96]
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "raw" / "port1_port2_water_content.csv"
RESULTS = ROOT / "results"


def load_series(column: str) -> pd.Series:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATA_PATH}. Run download_data.py from this folder first."
        )
    frame = pd.read_csv(DATA_PATH, sep=";")
    frame["datetime"] = pd.to_datetime(frame["datetime"], dayfirst=True)
    frame = frame.set_index("datetime").sort_index()
    # A regular 15-minute grid makes shift(N) mean N intervals, including gaps.
    series = frame[column].astype(float).resample("15min").mean()
    return series.rename(column)


def build_frame(series: pd.Series, horizon: int) -> pd.DataFrame:
    frame = pd.DataFrame(index=series.index)
    frame["target"] = series.shift(-horizon)
    for lag in LAGS:
        frame[f"lag_{lag}"] = series.shift(lag)
    # Rolling windows end at the current timestamp. Future readings are not included.
    gap_count = series.isna().astype(int)
    frame["mean_last_4h"] = series.rolling(16, min_periods=16).mean()
    frame["mean_last_24h"] = series.rolling(96, min_periods=96).mean()
    frame.loc[gap_count.rolling(16, min_periods=16).sum() > 0, "mean_last_4h"] = np.nan
    frame.loc[gap_count.rolling(96, min_periods=96).sum() > 0, "mean_last_24h"] = np.nan
    # Yesterday at the clock time we are forecasting. Known now when horizon <= 24h.
    frame["yesterday_at_target_clock"] = series.shift(96 - horizon)
    hour = series.index.hour
    day = series.index.dayofyear
    frame["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    frame["doy_sin"] = np.sin(2 * np.pi * day / 365)
    frame["doy_cos"] = np.cos(2 * np.pi * day / 365)
    return frame.dropna()


def time_split(frame: pd.DataFrame, start, end):
    cutoff = start + 0.8 * (end - start)
    train = frame.loc[frame.index < cutoff]
    test = frame.loc[frame.index >= cutoff]
    return train, test, cutoff


def regression_scores(y_true, y_pred) -> dict:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def feature_columns() -> list[str]:
    return (
        [f"lag_{lag}" for lag in LAGS]
        + ["mean_last_4h", "mean_last_24h", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    )


def fit_models(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    columns = feature_columns()
    x_train = train[columns]
    y_train = train["target"]
    x_test = test[columns]
    y_test = test["target"]
    models = {
        "linear_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=5,
            random_state=SEED,
        ),
        "svm_linear": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearSVR(max_iter=3000, random_state=SEED)),
            ]
        ),
    }
    rows = []
    stored_predictions = {
        "persistence": test["lag_0"].to_numpy(),
        "yesterday": test["yesterday_at_target_clock"].to_numpy(),
    }
    for name, prediction in stored_predictions.items():
        scores = regression_scores(y_test, prediction)
        scores["model"] = name
        rows.append(scores)
    fitted = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        scores = regression_scores(y_test, prediction)
        scores["model"] = name
        rows.append(scores)
        fitted[name] = model
        stored_predictions[name] = prediction
    return pd.DataFrame(rows), stored_predictions, fitted


def save_forecast_plot(test: pd.DataFrame, prediction: np.ndarray, path: Path, title: str) -> None:
    window = test.iloc[-96 * 14 :]
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.plot(window.index, window["target"], label="Recorded", color="#1f2933", linewidth=1.2)
    ax.plot(window.index, window["lag_0"].iloc[-len(window) :], label="Persistence", color="#9aa5b1", linewidth=1)
    aligned = pd.Series(prediction, index=test.index).loc[window.index]
    ax.plot(aligned.index, aligned, label="Random forest", color="#3b6d8f", linewidth=1.2)
    ax.set_ylabel("Soil water content")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_residual_plot(y_true, y_pred, path: Path) -> None:
    residual = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.hist(residual, bins=40, color="#3b6d8f")
    ax.set_xlabel("Recorded − predicted water content")
    ax.set_ylabel("Test timestamps")
    ax.set_title("4-hour residuals — port 1, random forest")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def gap_report(series: pd.Series) -> dict:
    missing = series.isna()
    # After resampling, the index itself is regular. Gaps show up as missing values.
    run_id = missing.ne(missing.shift(fill_value=False)).cumsum()
    longest_missing_steps = 0
    missing_stretches = 0
    for _, run in missing.groupby(run_id):
        if bool(run.iloc[0]):
            missing_stretches += 1
            longest_missing_steps = max(longest_missing_steps, int(len(run)))
    return {
        "start": str(series.index.min()),
        "end": str(series.index.max()),
        "grid_points": int(len(series)),
        "missing_grid_points": int(missing.sum()),
        "missing_stretches": missing_stretches,
        "longest_missing_minutes": int(longest_missing_steps * 15),
        "observed_min": float(series.min()),
        "observed_max": float(series.max()),
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    port1 = load_series("port1_water_content")
    port2 = load_series("port2_water_content")
    audit = {"port1": gap_report(port1), "port2": gap_report(port2)}
    start, end = port1.index.min(), port1.index.max()

    all_rows = []
    headline_predictions = None
    headline_test = None
    headline_forest = None
    headline_cutoff = None
    headline_train = None
    for horizon_name, horizon in HORIZONS.items():
        frame = build_frame(port1, horizon)
        train, test, cutoff = time_split(frame, start, end)
        scores, predictions, fitted = fit_models(train, test)
        scores.insert(0, "port", "port1_30cm_odd")
        scores.insert(1, "horizon", horizon_name)
        scores.insert(2, "horizon_steps", horizon)
        all_rows.append(scores)
        print(horizon_name)
        print(scores[["model", "mae", "rmse", "r2"]].to_string(index=False))
        if horizon == HEADLINE_HORIZON:
            headline_predictions = predictions
            headline_test = test
            headline_forest = fitted["random_forest"]
            headline_cutoff = cutoff
            headline_train = train

    comparison = pd.concat(all_rows, ignore_index=True)
    comparison.to_csv(RESULTS / "model_comparison.csv", index=False)

    columns = feature_columns()
    importance = pd.DataFrame(
        {"feature": columns, "importance": headline_forest.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance.to_csv(RESULTS / "random_forest_importance.csv", index=False)

    save_forecast_plot(
        headline_test,
        headline_predictions["random_forest"],
        RESULTS / "forecast_last_14_test_days.png",
        "Port 1 — last 14 days of the test period, 4-hour horizon",
    )
    save_residual_plot(
        headline_test["target"],
        headline_predictions["random_forest"],
        RESULTS / "residual_histogram.png",
    )

    # Same 4-hour protocol on the deeper sensor. Not a second research dataset.
    frame2 = build_frame(port2, HEADLINE_HORIZON)
    train2, test2, _ = time_split(frame2, start, end)
    port2_scores, _, _ = fit_models(train2, test2)
    port2_scores.insert(0, "port", "port2_60cm_even")
    port2_scores.insert(1, "horizon", "4_hour")
    port2_scores.to_csv(RESULTS / "port2_4h_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    headline = comparison.loc[comparison["horizon"] == "4_hour"]
    ax.bar(headline["model"], headline["mae"], color="#3b6d8f")
    ax.set_ylabel("MAE (soil water content)")
    ax.set_title("Port 1, 4-hour horizon — lower is better")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(RESULTS / "mae_comparison_4h.png", dpi=120)
    plt.close(fig)

    import sklearn

    summary = {
        "seed": SEED,
        "series": "port1_water_content from z6-03235_port1-2_water_content.csv",
        "depth_note": "Zenodo text: odd ports at 30 cm, even ports at 60 cm.",
        "headline_horizon": "4 hours (16 steps of 15 minutes)",
        "split": "first 80% of the calendar span for training, last 20% for test",
        "cutoff": str(headline_cutoff),
        "train_rows_4h": int(len(headline_train)),
        "test_rows_4h": int(len(headline_test)),
        "features": columns,
        "arima": "not fit; see module README",
        "rainfall_and_irrigation_files": "present in the Zenodo zip, not used as inputs in this baseline",
        "audit": audit,
        "packages": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
        },
    }
    (RESULTS / "run_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
