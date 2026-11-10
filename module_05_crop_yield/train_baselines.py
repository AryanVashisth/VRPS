"""Module 5 — predict recorded crop yield for a future year.

Split: train on 1997–2016, test on 2017–2020. Rows are not shuffled.
2020 has very few rows in this file; that is reported, not hidden.

Production is not used as an input. In this table, Yield is approximately
Production / Area, so Production would leak the answer.

Fertilizer and Pesticide are left out of the main models. Across the file,
Fertilizer / Area takes only a handful of values and those values follow the
year. They behave like a year-level rate times area, not like a soil test.
A separate random forest puts them back in, so the difference can be measured.

The main scores keep every test row, including extreme yields. Median absolute
error is reported beside RMSE because a few impossible-looking yields dominate
RMSE. Those rows are listed; they are not deleted to improve the score.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor

SEED = 42
TRAIN_LAST_YEAR = 2016
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "raw" / "crop_yield.csv"
RESULTS = ROOT / "results"

# Bajra is pearl millet in this table. The other names are related grains,
# not pearl millet, and are reported separately.
PEARL_MILLET = "Bajra"
RELATED_MILLETS = ["Bajra", "Small millets", "Jowar", "Ragi"]

NUMERIC = ["Crop_Year", "Area", "Annual_Rainfall"]
CATEGORICAL = ["Crop", "Season", "State"]
EXTRA_RATE_FEATURES = ["Fertilizer", "Pesticide"]
TARGET = "Yield"


def load_table() -> tuple[pd.DataFrame, dict]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATA_PATH}. Run download_data.py from this folder first."
        )
    frame = pd.read_csv(DATA_PATH)
    for column in CATEGORICAL:
        frame[column] = frame[column].astype(str).str.strip()
    for column in NUMERIC + EXTRA_RATE_FEATURES + ["Production", TARGET]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    notes = {
        "rows_loaded": int(len(frame)),
        "rows_missing_yield_or_features": int(
            frame[NUMERIC + CATEGORICAL + [TARGET]].isna().any(axis=1).sum()
        ),
        "rows_with_non_positive_area": int((frame["Area"] <= 0).sum()),
    }
    frame = frame.dropna(subset=NUMERIC + CATEGORICAL + [TARGET])
    frame = frame.loc[frame["Area"] > 0].copy()
    frame["production_over_area"] = frame["Production"] / frame["Area"]
    ratio = frame["production_over_area"] / frame[TARGET].replace(0, np.nan)
    notes["median_production_over_area_divided_by_yield"] = float(ratio.median())
    notes["rows_used"] = int(len(frame))
    notes["year_counts"] = {str(k): int(v) for k, v in frame["Crop_Year"].value_counts().sort_index().items()}
    return frame, notes


def regression_scores(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "median_ae": median_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def make_pipeline(model, numeric_features: list[str], scale: bool) -> Pipeline:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric = Pipeline(numeric_steps)
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocess = ColumnTransformer(
        [
            ("num", numeric, numeric_features),
            ("cat", categorical, CATEGORICAL),
        ]
    )
    return Pipeline([("prep", preprocess), ("model", model)])


def candidate_models() -> dict:
    return {
        "linear_regression": (
            LinearRegression(),
            True,
        ),
        "decision_tree": (
            DecisionTreeRegressor(max_depth=8, min_samples_leaf=10, random_state=SEED),
            False,
        ),
        "random_forest": (
            RandomForestRegressor(
                n_estimators=200,
                max_depth=16,
                min_samples_leaf=5,
                random_state=SEED,
            ),
            False,
        ),
        "gradient_boosting": (
            GradientBoostingRegressor(
                n_estimators=150, learning_rate=0.08, max_depth=3, random_state=SEED
            ),
            False,
        ),
        "svm_linear": (
            LinearSVR(max_iter=4000, random_state=SEED),
            True,
        ),
    }


def crop_mean_predict(train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> np.ndarray:
    means = train_frame.groupby("Crop")[TARGET].mean()
    fallback = float(train_frame[TARGET].mean())
    return test_frame["Crop"].map(means).fillna(fallback).to_numpy()


def evaluate_slice(name: str, y_true, y_pred, mask: pd.Series) -> dict:
    scores = regression_scores(y_true[mask], y_pred[mask])
    scores["slice"] = name
    scores["rows"] = int(mask.sum())
    return scores


def save_residual_plots(y_true, y_pred, path_scatter: Path, path_hist: Path, title: str) -> None:
    residual = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    ax.scatter(y_pred, y_true, s=8, alpha=0.35, color="#3b6d8f")
    low = min(np.min(y_pred), np.min(y_true))
    high = max(np.max(y_pred), np.max(y_true))
    ax.plot([low, high], [low, high], color="black", linewidth=1)
    ax.set_xlabel("Predicted yield")
    ax.set_ylabel("Recorded yield")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path_scatter, dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.hist(residual, bins=40, color="#3b6d8f")
    ax.set_xlabel("Recorded − predicted")
    ax.set_ylabel("Test rows")
    ax.set_title("Yield residuals")
    fig.tight_layout()
    fig.savefig(path_hist, dpi=120)
    plt.close(fig)


def save_importance(pipeline: Pipeline, numeric_features: list[str], path_csv: Path, path_png: Path) -> None:
    names = [name.split("__", 1)[-1] for name in pipeline.named_steps["prep"].get_feature_names_out()]
    table = pd.DataFrame(
        {"feature": names, "importance": pipeline.named_steps["model"].feature_importances_}
    ).sort_values("importance", ascending=False)
    table.to_csv(path_csv, index=False)
    top = table.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.barh(top["feature"], top["importance"], color="#3b6d8f")
    ax.set_xlabel("Random Forest importance (impurity decrease)")
    ax.set_title("Yield model — main feature set")
    fig.tight_layout()
    fig.savefig(path_png, dpi=120)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame, notes = load_table()
    train = frame.loc[frame["Crop_Year"] <= TRAIN_LAST_YEAR].copy()
    test = frame.loc[frame["Crop_Year"] > TRAIN_LAST_YEAR].copy()
    y_train = train[TARGET]
    y_test = test[TARGET]

    rows = []
    mean_prediction = np.full(len(test), float(y_train.mean()))
    mean_scores = regression_scores(y_test, mean_prediction)
    mean_scores["model"] = "train_mean_baseline"
    mean_scores["feature_set"] = "none"
    rows.append(mean_scores)

    crop_prediction = crop_mean_predict(train, test)
    crop_scores = regression_scores(y_test, crop_prediction)
    crop_scores["model"] = "crop_mean_baseline"
    crop_scores["feature_set"] = "Crop only"
    rows.append(crop_scores)

    fitted = {}
    predictions = {"crop_mean_baseline": crop_prediction}
    for name, (model, scale) in candidate_models().items():
        pipeline = make_pipeline(model, NUMERIC, scale)
        pipeline.fit(train[NUMERIC + CATEGORICAL], y_train)
        prediction = pipeline.predict(test[NUMERIC + CATEGORICAL])
        scores = regression_scores(y_test, prediction)
        scores["model"] = name
        scores["feature_set"] = "crop_season_state_year_area_rainfall"
        rows.append(scores)
        fitted[name] = pipeline
        predictions[name] = prediction
        print(name, {key: round(scores[key], 3) for key in ("mae", "rmse", "r2")})

    # Same random forest, plus the two rate columns, on the same year split.
    rate_features = NUMERIC + EXTRA_RATE_FEATURES
    rate_forest = make_pipeline(candidate_models()["random_forest"][0], rate_features, False)
    rate_columns = rate_features + CATEGORICAL
    rate_forest.fit(train[rate_columns], y_train)
    rate_prediction = rate_forest.predict(test[rate_columns])
    rate_scores = regression_scores(y_test, rate_prediction)
    rate_scores["model"] = "random_forest"
    rate_scores["feature_set"] = "main_plus_fertilizer_and_pesticide"
    rows.append(rate_scores)

    comparison = pd.DataFrame(rows)
    comparison = comparison[
        ["feature_set", "model", "mae", "rmse", "median_ae", "r2"]
    ].sort_values(["feature_set", "mae"])
    comparison.to_csv(RESULTS / "model_comparison.csv", index=False)

    main_models = comparison.loc[
        comparison["feature_set"] == "crop_season_state_year_area_rainfall"
    ].sort_values("mae")
    best_name = main_models.iloc[0]["model"]
    best_prediction = predictions[best_name]
    save_residual_plots(
        y_test,
        best_prediction,
        RESULTS / "residual_scatter.png",
        RESULTS / "residual_histogram.png",
        f"Test years 2017–2020 — {best_name}",
    )
    save_importance(
        fitted["random_forest"],
        NUMERIC,
        RESULTS / "random_forest_importance.csv",
        RESULTS / "random_forest_importance.png",
    )

    slice_rows = []
    for model_name, prediction in {
        "crop_mean_baseline": crop_prediction,
        best_name: best_prediction,
        "random_forest": predictions["random_forest"],
    }.items():
        slice_rows.append(evaluate_slice(f"{model_name} | all test rows", y_test, prediction, pd.Series(True, index=test.index)))
        slice_rows.append(
            evaluate_slice(
                f"{model_name} | Bajra only",
                y_test,
                prediction,
                test["Crop"] == PEARL_MILLET,
            )
        )
        slice_rows.append(
            evaluate_slice(
                f"{model_name} | Bajra, small millets, jowar, ragi",
                y_test,
                prediction,
                test["Crop"].isin(RELATED_MILLETS),
            )
        )
        slice_rows.append(
            evaluate_slice(
                f"{model_name} | all test rows except Coconut",
                y_test,
                prediction,
                test["Crop"] != "Coconut",
            )
        )
    pd.DataFrame(slice_rows).to_csv(RESULTS / "millet_slice_metrics.csv", index=False)

    extremes = frame.nlargest(15, TARGET)[
        ["Crop", "Crop_Year", "Season", "State", "Area", "Production", "Yield"]
    ]
    extremes.to_csv(RESULTS / "largest_recorded_yields.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    plot_rows = comparison.loc[comparison["feature_set"] != "main_plus_fertilizer_and_pesticide"]
    ax.bar(plot_rows["model"], plot_rows["mae"], color="#3b6d8f")
    ax.set_ylabel("MAE on 2017–2020")
    ax.set_title("Yield models — lower is better")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(RESULTS / "mae_comparison.png", dpi=120)
    plt.close(fig)

    import sklearn

    summary = {
        "seed": SEED,
        "split": f"Crop_Year <= {TRAIN_LAST_YEAR} train; later years test",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "test_rows_in_2020": int((test["Crop_Year"] == 2020).sum()),
        "excluded_because_it_reconstructs_yield": ["Production"],
        "left_out_of_main_models": EXTRA_RATE_FEATURES,
        "main_features": NUMERIC + CATEGORICAL,
        "best_main_model_by_mae": best_name,
        "best_main_mae": float(main_models.iloc[0]["mae"]),
        "selection_note": (
            "Lowest all-crop MAE is gradient boosting. That lead comes from Coconut, "
            "whose recorded yield is orders of magnitude above the grain crops. "
            "Random forest is the model carried forward: on non-coconut test rows, "
            "and on Bajra, it has the lower error."
        ),
        "audit": notes,
        "packages": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
        },
    }
    (RESULTS / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(comparison.to_string(index=False))
    print("slices")
    print(pd.DataFrame(slice_rows).to_string(index=False))


if __name__ == "__main__":
    main()
