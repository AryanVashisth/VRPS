"""Module 1 — predict the documented soil-fertility class.

The target already exists in the file. It is not invented here.
Class names follow the dataset author's README:

    0 = Less Fertile, 1 = Fertile, 2 = Highly Fertile

The published file repeats the Highly Fertile rows. Those copies are removed
before the real comparison. A second run keeps the copies on purpose, so the
results folder can show how duplicate rows inflate a random split.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
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
DATA_PATH = ROOT / "data" / "raw" / "soil_fertility.csv"
RESULTS = ROOT / "results"

FEATURES = ["N", "P", "K", "ph", "ec", "oc", "S", "zn", "fe", "cu", "Mn", "B"]
TARGET = "fertility"
CLASS_NAMES = {
    0: "Less Fertile",
    1: "Fertile",
    2: "Highly Fertile",
}
CLASS_ORDER = ["Less Fertile", "Fertile", "Highly Fertile"]


def load_table() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATA_PATH}. Run download_data.py from this folder first."
        )
    frame = pd.read_csv(DATA_PATH)
    missing = [column for column in FEATURES + [TARGET] if column not in frame.columns]
    if missing:
        raise ValueError(f"Unexpected columns. Missing {missing}. Found {list(frame.columns)}")
    frame = frame[FEATURES + [TARGET]].copy()
    frame[TARGET] = frame[TARGET].map(CLASS_NAMES)
    unknown = sorted(set(frame[TARGET].dropna()) - set(CLASS_ORDER))
    if frame[TARGET].isna().any() or unknown:
        raise ValueError(f"Target contains labels outside the documented map: {unknown}")
    return frame


def score_predictions(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def make_models() -> dict:
    # min_samples_leaf stops a tree from building a rule on one or two rows.
    # class_weight keeps the rare Highly Fertile class visible to the model.
    return {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "decision_tree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=5, class_weight="balanced", random_state=SEED
        ),
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=SEED
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=SEED,
        ),
        "svm_rbf": SVC(kernel="rbf", C=1.0, class_weight="balanced", random_state=SEED),
    }


def make_pipeline(model) -> Pipeline:
    # Scaling is fit on the training fold only, because it sits inside the pipeline.
    # Trees do not need scaling, but a shared pipeline keeps the comparison fair.
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocess = ColumnTransformer([("num", numeric, FEATURES)])
    return Pipeline([("prep", preprocess), ("model", model)])


def duplicate_overlap(train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> int:
    train_keys = set(map(tuple, train_frame.to_numpy()))
    return sum(tuple(row) in train_keys for row in test_frame.to_numpy())


def save_confusion(y_true, y_pred, path: Path, title: str) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(CLASS_ORDER)), CLASS_ORDER, rotation=25, ha="right")
    ax.set_yticks(range(len(CLASS_ORDER)), CLASS_ORDER)
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


def save_metric_bars(scores: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(scores["model"], scores["macro_f1"], color="#3b6d8f")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro F1")
    ax.set_title("Soil fertility — same split, deduplicated rows")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_importance(pipeline: Pipeline, path_csv: Path, path_png: Path) -> None:
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return
    names = pipeline.named_steps["prep"].get_feature_names_out()
    # ColumnTransformer prefixes names with "num__".
    clean_names = [name.split("__", 1)[-1] for name in names]
    table = pd.DataFrame(
        {"feature": clean_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    table.to_csv(path_csv, index=False)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.barh(table["feature"][::-1], table["importance"][::-1], color="#3b6d8f")
    ax.set_xlabel("Random Forest importance (impurity decrease)")
    ax.set_title("Which measurements the forest used")
    fig.tight_layout()
    fig.savefig(path_png, dpi=120)
    plt.close(fig)


def evaluate_frame(frame: pd.DataFrame, protocol: str) -> tuple[pd.DataFrame, dict]:
    features = frame[FEATURES]
    target = frame[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=target,
    )
    overlap = duplicate_overlap(
        pd.concat([x_train, y_train.rename(TARGET)], axis=1),
        pd.concat([x_test, y_test.rename(TARGET)], axis=1),
    )
    rows = []
    fitted = {}
    reports = {}
    for name, model in make_models().items():
        pipeline = make_pipeline(clone(model))
        pipeline.fit(x_train, y_train)
        prediction = pipeline.predict(x_test)
        row = score_predictions(y_test, prediction)
        row["model"] = name
        row["protocol"] = protocol
        rows.append(row)
        fitted[name] = (pipeline, prediction, y_test)
        reports[name] = classification_report(
            y_test, prediction, labels=CLASS_ORDER, zero_division=0
        )
    details = {
        "protocol": protocol,
        "rows": int(len(frame)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "class_counts": frame[TARGET].value_counts().to_dict(),
        "test_rows_copied_from_train": int(overlap),
        "reports": reports,
        "fitted": fitted,
    }
    return pd.DataFrame(rows), details


def range_audit(frame: pd.DataFrame) -> pd.DataFrame:
    audit = frame[FEATURES].agg(["min", "median", "max"]).T.reset_index(names="feature")
    audit["flag"] = ""
    if (frame["ph"] < 3).any() or (frame["ph"] > 11).any():
        audit.loc[audit["feature"] == "ph", "flag"] = "pH outside 3–11"
    for feature in FEATURES:
        if feature == "ph":
            continue
        if (frame[feature] < 0).any():
            audit.loc[audit["feature"] == feature, "flag"] = "negative value"
    return audit


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    raw = load_table()
    deduped = raw.drop_duplicates().reset_index(drop=True)

    raw_scores, raw_details = evaluate_frame(raw, "invalid_with_duplicate_rows")
    clean_scores, clean_details = evaluate_frame(deduped, "deduplicated_holdout")

    # The invalid run is stored so the inflation is visible. It is not a candidate.
    all_scores = pd.concat([clean_scores, raw_scores], ignore_index=True)
    all_scores = all_scores[
        ["protocol", "model", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]
    ]
    all_scores.to_csv(RESULTS / "model_comparison.csv", index=False)

    clean_only = clean_scores.sort_values("macro_f1", ascending=False)
    save_metric_bars(clean_only, RESULTS / "macro_f1_comparison.png")

    best_name = clean_only.iloc[0]["model"]
    best_pipeline, best_prediction, y_test = clean_details["fitted"][best_name]
    save_confusion(
        y_test,
        best_prediction,
        RESULTS / "confusion_matrix_best.png",
        f"Deduplicated test set — {best_name}",
    )
    forest = clean_details["fitted"]["random_forest"][0]
    save_importance(
        forest,
        RESULTS / "random_forest_importance.csv",
        RESULTS / "random_forest_importance.png",
    )

    for name, report in clean_details["reports"].items():
        (RESULTS / f"classification_report_{name}.txt").write_text(report)

    range_audit(deduped).to_csv(RESULTS / "feature_ranges.csv", index=False)
    balance = pd.DataFrame(
        {
            "class": CLASS_ORDER,
            "rows_in_published_file": [int((raw[TARGET] == name).sum()) for name in CLASS_ORDER],
            "rows_after_dropping_exact_copies": [
                int((deduped[TARGET] == name).sum()) for name in CLASS_ORDER
            ],
        }
    )
    balance.to_csv(RESULTS / "class_balance.csv", index=False)

    summary = {
        "seed": SEED,
        "test_size": TEST_SIZE,
        "published_rows": int(len(raw)),
        "unique_rows": int(len(deduped)),
        "exact_duplicate_rows_removed": int(len(raw) - len(deduped)),
        "features": FEATURES,
        "target": CLASS_NAMES,
        "invalid_protocol_test_rows_copied_from_train": raw_details["test_rows_copied_from_train"],
        "clean_protocol_test_rows_copied_from_train": clean_details["test_rows_copied_from_train"],
        "best_model_by_macro_f1": best_name,
        "best_macro_f1": float(clean_only.iloc[0]["macro_f1"]),
        "packages": {"pandas": pd.__version__},
    }
    import sklearn

    summary["packages"]["scikit-learn"] = sklearn.__version__
    (RESULTS / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(all_scores.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
