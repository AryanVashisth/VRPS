"""Module 6 — classify the crop label from N, P, K, climate and pH.

The assigned table has 22 crops and 100 rows each. It does not contain pearl
millet or any millet class. High accuracy on this file is a statement about
the table, not about a millet field.

After the comparison, the script holds a crop's other measurements at their
training median and changes only pH. That is the check requested by the
project's earlier pH-sensitivity note. A flip means pH alone changed the
recommended crop for that median profile. No flip means it did not, for the
profiles that were tested.
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
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 42
TEST_SIZE = 0.20
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "raw" / "crop_recommendation.csv"
RESULTS = ROOT / "results"

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"
# Crops used only for the pH probe. They are not a millet substitute.
PROBE_CROPS = ["rice", "maize", "cotton", "chickpea"]
PH_GRID = [4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5]


def load_table() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATA_PATH}. Run download_data.py from this folder first."
        )
    frame = pd.read_csv(DATA_PATH)
    missing = [column for column in FEATURES + [TARGET] if column not in frame.columns]
    if missing:
        raise ValueError(f"Unexpected columns. Missing {missing}")
    return frame[FEATURES + [TARGET]].copy()


def make_pipeline(model) -> Pipeline:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocess = ColumnTransformer([("num", numeric, FEATURES)])
    return Pipeline([("prep", preprocess), ("model", model)])


def make_models() -> dict:
    return {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "decision_tree": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=5, random_state=SEED
        ),
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=SEED),
        "random_forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, random_state=SEED
        ),
        "svm_rbf": SVC(kernel="rbf", C=1.0, random_state=SEED),
    }


def score_predictions(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def save_confusion(y_true, y_pred, labels, path: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(10.5, 9.0))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Crop recommendation — held-out 20%")
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_importance(pipeline: Pipeline) -> None:
    names = [name.split("__", 1)[-1] for name in pipeline.named_steps["prep"].get_feature_names_out()]
    table = pd.DataFrame(
        {"feature": names, "importance": pipeline.named_steps["model"].feature_importances_}
    ).sort_values("importance", ascending=False)
    table.to_csv(RESULTS / "random_forest_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ordered = table.iloc[::-1]
    ax.barh(ordered["feature"], ordered["importance"], color="#3b6d8f")
    ax.set_xlabel("Random Forest importance (impurity decrease)")
    ax.set_title("Crop recommendation — training fold")
    fig.tight_layout()
    fig.savefig(RESULTS / "random_forest_importance.png", dpi=120)
    plt.close(fig)


def ph_sensitivity(pipeline: Pipeline, train_frame: pd.DataFrame) -> pd.DataFrame:
    """Change only pH. Every other column stays at that crop's training median."""
    rows = []
    for crop in PROBE_CROPS:
        crop_rows = train_frame.loc[train_frame[TARGET] == crop, FEATURES]
        median_profile = crop_rows.median()
        observed_low = float(crop_rows["ph"].min())
        observed_high = float(crop_rows["ph"].max())
        for ph in PH_GRID:
            sample = median_profile.copy()
            sample["ph"] = ph
            predicted = pipeline.predict(pd.DataFrame([sample]))[0]
            rows.append(
                {
                    "crop_profile": crop,
                    "ph": ph,
                    "ph_inside_that_crops_training_range": observed_low <= ph <= observed_high,
                    "predicted_crop": predicted,
                    "prediction_matches_profile_crop": predicted == crop,
                    "profile_N": float(median_profile["N"]),
                    "profile_P": float(median_profile["P"]),
                    "profile_K": float(median_profile["K"]),
                    "profile_rainfall": float(median_profile["rainfall"]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame = load_table()
    class_counts = frame[TARGET].value_counts().sort_index()
    class_counts.to_csv(RESULTS / "class_balance.csv", header=["rows"])

    x_train, x_test, y_train, y_test = train_test_split(
        frame[FEATURES],
        frame[TARGET],
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=frame[TARGET],
    )
    labels = sorted(frame[TARGET].unique())
    rows = []
    fitted = {}
    for name, model in make_models().items():
        pipeline = make_pipeline(model)
        pipeline.fit(x_train, y_train)
        prediction = pipeline.predict(x_test)
        row = score_predictions(y_test, prediction)
        row["model"] = name
        rows.append(row)
        fitted[name] = (pipeline, prediction)
        (RESULTS / f"classification_report_{name}.txt").write_text(
            classification_report(y_test, prediction, labels=labels, zero_division=0)
        )
        print(name, round(row["macro_f1"], 3))

    comparison = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    comparison = comparison[
        ["model", "accuracy", "macro_precision", "macro_recall", "macro_f1"]
    ]
    comparison.to_csv(RESULTS / "model_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(comparison["model"], comparison["macro_f1"], color="#3b6d8f")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro F1")
    ax.set_title("Crop recommendation — same stratified 80/20 split")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(RESULTS / "macro_f1_comparison.png", dpi=120)
    plt.close(fig)

    best_name = comparison.iloc[0]["model"]
    save_confusion(y_test, fitted[best_name][1], labels, RESULTS / "confusion_matrix_best.png")

    forest = fitted["random_forest"][0]
    save_importance(forest)

    train_frame = x_train.copy()
    train_frame[TARGET] = y_train.to_numpy()
    sensitivity = ph_sensitivity(forest, train_frame)
    sensitivity.to_csv(RESULTS / "ph_sensitivity.csv", index=False)
    flips = (
        sensitivity.groupby("crop_profile")["prediction_matches_profile_crop"]
        .apply(lambda values: int((~values).sum()))
        .rename("ph_values_that_changed_the_crop")
    )
    flips.to_csv(RESULTS / "ph_sensitivity_flip_counts.csv")

    import sklearn

    summary = {
        "seed": SEED,
        "test_size": TEST_SIZE,
        "rows": int(len(frame)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "n_classes": int(frame[TARGET].nunique()),
        "classes": labels,
        "duplicate_rows": int(frame.duplicated().sum()),
        "millet_classes_present": False,
        "best_model_by_macro_f1": best_name,
        "best_macro_f1": float(comparison.iloc[0]["macro_f1"]),
        "ph_probe_crops": PROBE_CROPS,
        "ph_grid": PH_GRID,
        "packages": {"pandas": pd.__version__, "numpy": np.__version__, "scikit-learn": sklearn.__version__},
    }
    (RESULTS / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(comparison.to_string(index=False))
    print(flips.to_string())


if __name__ == "__main__":
    main()
