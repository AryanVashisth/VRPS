"""Module 3 — classical baseline for pearl-millet leaf images.

This script does not train a neural net. It resizes each leaf to 64x64 and
keeps a small set of colour summaries: channel mean, channel standard
deviation, and an 8-bin histogram per channel. A random forest on those
numbers is easy to rerun and gives the tabular-style comparison the other
modules use.

The two large Zenodo archives (smut, about 230 MB, and Pyricularia, about
370 MB) are not downloaded. This folder keeps the three small classes so the
module fits on a normal PC.

The Zenodo folders do not name the plant or the leaf. A random split can
therefore put two pictures of the same leaf on both sides. Treat the score
as optimistic until a plant-level split exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

SEED = 42
TEST_SIZE = 0.20
ROOT = Path(__file__).resolve().parent
IMAGE_ROOT = ROOT / "data" / "images"
RESULTS = ROOT / "results"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images() -> pd.DataFrame:
    if not IMAGE_ROOT.exists():
        raise FileNotFoundError(
            f"Missing {IMAGE_ROOT}. Run download_data.py and wait for the archives to finish."
        )
    rows = []
    for class_dir in sorted(path for path in IMAGE_ROOT.iterdir() if path.is_dir()):
        for path in class_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                rows.append({"path": str(path), "label": class_dir.name})
    if not rows:
        raise FileNotFoundError(f"No images found under {IMAGE_ROOT}.")
    return pd.DataFrame(rows)


def colour_features(path: str) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((64, 64))
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    means = pixels.mean(axis=(0, 1))
    stds = pixels.std(axis=(0, 1))
    histograms = []
    for channel in range(3):
        counts, _ = np.histogram(pixels[:, :, channel], bins=8, range=(0.0, 1.0))
        histograms.append(counts / counts.sum())
    return np.concatenate([means, stds, *histograms])


def feature_names() -> list[str]:
    names = ["mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b"]
    for channel in ("r", "g", "b"):
        names.extend([f"hist_{channel}_{bin_id}" for bin_id in range(8)])
    return names


def score_predictions(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def save_confusion(y_true, y_pred, labels, path: Path, title: str) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
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


def record_mistakes(frame: pd.DataFrame, y_true, y_pred) -> None:
    """Write filenames of a few mistakes. The image files stay where they are."""
    wrong = frame.loc[np.asarray(y_true) != np.asarray(y_pred)].copy()
    wrong["predicted"] = np.asarray(y_pred)[np.asarray(y_true) != np.asarray(y_pred)]
    kept = []
    for label, group in wrong.groupby("label"):
        for _, example in group.head(2).iterrows():
            kept.append(
                {
                    "true_label": label,
                    "predicted_label": example["predicted"],
                    "image": Path(example["path"]).name,
                }
            )
    pd.DataFrame(kept).to_csv(RESULTS / "misclassified_examples.csv", index=False)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    index = list_images()
    counts = index["label"].value_counts().sort_index()
    counts.to_csv(RESULTS / "class_balance.csv", header=["images"])
    print("images", len(index))
    print(counts.to_string())

    features = np.vstack([colour_features(path) for path in index["path"]])
    feature_frame = pd.DataFrame(features, columns=feature_names())
    labels = sorted(index["label"].unique())

    x_train, x_test, y_train, y_test, index_train, index_test = train_test_split(
        feature_frame,
        index["label"],
        index,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=index["label"],
    )
    models = {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "decision_tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=5, class_weight="balanced", random_state=SEED
        ),
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=SEED,
        ),
        "svm_linear": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearSVC(class_weight="balanced", random_state=SEED, max_iter=3000)),
            ]
        ),
    }

    rows = []
    predictions = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        row = score_predictions(y_test, prediction)
        row["model"] = name
        rows.append(row)
        predictions[name] = prediction
        (RESULTS / f"classification_report_{name}.txt").write_text(
            classification_report(y_test, prediction, labels=labels, zero_division=0)
        )
        print(name, round(row["macro_f1"], 3))

    comparison = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    comparison = comparison[["model", "accuracy", "macro_precision", "macro_recall", "macro_f1"]]
    comparison.to_csv(RESULTS / "model_comparison.csv", index=False)

    best_name = comparison.iloc[0]["model"]
    save_confusion(
        y_test,
        predictions[best_name],
        labels,
        RESULTS / "confusion_matrix_best.png",
        f"Colour-feature test set — {best_name}",
    )
    forest = models["random_forest"]
    importance = pd.DataFrame(
        {"feature": feature_names(), "importance": forest.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance.to_csv(RESULTS / "random_forest_importance.csv", index=False)
    top = importance.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.barh(top["feature"], top["importance"], color="#3b6d8f")
    ax.set_xlabel("Random Forest importance")
    ax.set_title("Leaf colour features the forest used")
    fig.tight_layout()
    fig.savefig(RESULTS / "random_forest_importance.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(comparison["model"], comparison["macro_f1"], color="#3b6d8f")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro F1")
    ax.set_title("Pearl millet leaves — colour features, same split")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(RESULTS / "macro_f1_comparison.png", dpi=120)
    plt.close(fig)

    record_mistakes(index_test.reset_index(drop=True), y_test.to_numpy(), predictions[best_name])

    import sklearn

    summary = {
        "seed": SEED,
        "classes_used": "healthy, sclerospora_graminicola, exserohilum_rostratum",
        "omitted_for_size": ["moesziomyces_bullatus (~230 MB)", "pyricularia (~370 MB)"],
        "test_size": TEST_SIZE,
        "images": int(len(index)),
        "train_images": int(len(x_train)),
        "test_images": int(len(x_test)),
        "class_counts": {label: int(count) for label, count in counts.items()},
        "features": "64x64 colour mean, standard deviation, and 8-bin histogram per channel",
        "split_warning": "Random image split. Plant or leaf identity is not in the Zenodo folders.",
        "augmentation": "none in this script",
        "best_model_by_macro_f1": best_name,
        "best_macro_f1": float(comparison.iloc[0]["macro_f1"]),
        "packages": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
        },
    }
    (RESULTS / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
