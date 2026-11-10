"""Module 4 — predict the fertilizer name from soil, crop and weather columns.

The assigned file has two problems that are fixed before any score is trusted:

1. Every row is stored twice. Exact copies are removed so a test fold cannot
   see the same recommendation it trained on.
2. The grade 10-26-26 was saved as the date 10/26/2026. Spreadsheet software
   often reads that grade as a date. It is mapped back to 10-26-26.

There are 99 recommendations after cleaning. A single 80/20 split would leave
one or two test rows in the smallest classes, so every model is scored with
the same 5-fold stratified split. The confusion matrix uses out-of-fold
predictions from that split.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

SEED = 42
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "raw" / "fertilizer_prediction.csv"
RESULTS = ROOT / "results"

# Source spelling "Temparature" is renamed so the rest of the script is readable.
# The rename is recorded in the run summary.
NUMERIC = ["Temperature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous"]
CATEGORICAL = ["Soil Type", "Crop Type"]
TARGET = "Fertilizer Name"
DATE_CORRUPTED_LABEL = "10/26/2026"
CORRECT_LABEL = "10-26-26"


def load_and_clean() -> tuple[pd.DataFrame, dict]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATA_PATH}. Run download_data.py from this folder first."
        )
    raw = pd.read_csv(DATA_PATH)
    raw.columns = [column.strip() for column in raw.columns]
    raw = raw.rename(columns={"Temparature": "Temperature"})
    notes = {
        "rows_in_file": int(len(raw)),
        "label_before_fix": raw[TARGET].value_counts().to_dict(),
    }
    raw[TARGET] = raw[TARGET].replace({DATE_CORRUPTED_LABEL: CORRECT_LABEL})
    cleaned = raw.drop_duplicates().reset_index(drop=True)
    notes["rows_after_dropping_exact_copies"] = int(len(cleaned))
    notes["label_after_fix"] = cleaned[TARGET].value_counts().to_dict()
    expected = set(NUMERIC + CATEGORICAL + [TARGET])
    if set(cleaned.columns) != expected:
        raise ValueError(f"Unexpected columns: {list(cleaned.columns)}")
    return cleaned, notes


def make_pipeline(model) -> Pipeline:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocess = ColumnTransformer(
        [
            ("num", numeric, NUMERIC),
            ("cat", categorical, CATEGORICAL),
        ]
    )
    return Pipeline([("prep", preprocess), ("model", model)])


def make_models() -> dict:
    return {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "decision_tree": DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=3, class_weight="balanced", random_state=SEED
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=SEED,
        ),
        "svm_rbf": SVC(kernel="rbf", C=1.0, class_weight="balanced", random_state=SEED),
        # Gradient boosting has no class_weight argument in scikit-learn.
        # Depth is kept small because several classes have only 7 rows.
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.08, max_depth=2, random_state=SEED
        ),
    }


def save_confusion(y_true, y_pred, labels, path: Path, title: str) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, int(matrix[row, col]), ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_importance(pipeline: Pipeline, path_csv: Path, path_png: Path) -> None:
    model = pipeline.named_steps["model"]
    names = [name.split("__", 1)[-1] for name in pipeline.named_steps["prep"].get_feature_names_out()]
    table = pd.DataFrame(
        {"feature": names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    table.to_csv(path_csv, index=False)
    top = table.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.barh(top["feature"], top["importance"], color="#3b6d8f")
    ax.set_xlabel("Random Forest importance (impurity decrease)")
    ax.set_title("Fertilizer model — descriptive fit on all 99 rows")
    fig.tight_layout()
    fig.savefig(path_png, dpi=120)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame, notes = load_and_clean()
    features = frame[NUMERIC + CATEGORICAL]
    target = frame[TARGET]
    labels = sorted(target.unique())
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = {
        "accuracy": "accuracy",
        "macro_precision": "precision_macro",
        "macro_recall": "recall_macro",
        "macro_f1": "f1_macro",
    }

    rows = []
    best_name = None
    best_f1 = -1.0
    best_oof = None
    for name, model in make_models().items():
        pipeline = make_pipeline(model)
        cv_scores = cross_validate(pipeline, features, target, cv=cv, scoring=scoring)
        oof = cross_val_predict(pipeline, features, target, cv=cv)
        row = {
            "model": name,
            "accuracy_mean": float(cv_scores["test_accuracy"].mean()),
            "accuracy_std": float(cv_scores["test_accuracy"].std()),
            "macro_precision_mean": float(cv_scores["test_macro_precision"].mean()),
            "macro_recall_mean": float(cv_scores["test_macro_recall"].mean()),
            "macro_f1_mean": float(cv_scores["test_macro_f1"].mean()),
            "macro_f1_std": float(cv_scores["test_macro_f1"].std()),
        }
        rows.append(row)
        (RESULTS / f"classification_report_{name}.txt").write_text(
            classification_report(target, oof, labels=labels, zero_division=0)
        )
        if row["macro_f1_mean"] > best_f1:
            best_f1 = row["macro_f1_mean"]
            best_name = name
            best_oof = oof
        print(name, round(row["macro_f1_mean"], 3), "±", round(row["macro_f1_std"], 3))

    comparison = pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False)
    comparison.to_csv(RESULTS / "model_comparison.csv", index=False)
    save_confusion(
        target,
        best_oof,
        labels,
        RESULTS / "confusion_matrix_best.png",
        f"Out-of-fold predictions — {best_name}",
    )

    # Importance is descriptive. The scores above do not use this full-data fit.
    forest = make_pipeline(make_models()["random_forest"])
    forest.fit(features, target)
    save_importance(
        forest,
        RESULTS / "random_forest_importance.csv",
        RESULTS / "random_forest_importance.png",
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(comparison["model"], comparison["macro_f1_mean"], color="#3b6d8f")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro F1 (5-fold mean)")
    ax.set_title("Fertilizer recommendation — same folds for every model")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(RESULTS / "macro_f1_comparison.png", dpi=120)
    plt.close(fig)

    import sklearn

    summary = {
        "seed": SEED,
        "split": "StratifiedKFold n_splits=5 shuffle=True",
        "rows_used": int(len(frame)),
        "cleaning": notes,
        "label_fix": {DATE_CORRUPTED_LABEL: CORRECT_LABEL},
        "features_numeric": NUMERIC,
        "features_categorical": CATEGORICAL,
        "best_model_by_macro_f1": best_name,
        "best_macro_f1_mean": best_f1,
        "packages": {"pandas": pd.__version__, "scikit-learn": sklearn.__version__},
        "not_used": "Kaggle playground synthetic fertilizer data was not merged into this file.",
    }
    (RESULTS / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
