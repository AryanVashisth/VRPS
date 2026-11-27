"""Rebuild Module 6 pilot: features, RandomForest pipeline, ONNX, metrics.

Reads images from the development module (absolute path below, override with
DEV_IMAGES env var). If that folder is missing, falls back to the copied
feature CSV in data/processed.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import colour_features, feature_names

ROOT = Path(__file__).resolve().parents[1]
PROC, ART, RES, DEMO = (ROOT / "data/processed", ROOT / "artifacts", ROOT / "results", ROOT / "data/demo_images")
DEV_IMAGES = Path(os.environ.get(
    "DEV_IMAGES",
    "/Users/aryan/Desktop/VRPS- Research Associate/development/module_03_pearl_millet_disease/data/images",
))
CSV_PATH = PROC / "pearl_millet_colour_features_v1.csv"
MODULE_ID = "module_06_pearl_millet_disease"
VERSION = "1.0.0"
SEED = 42
SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEMO_PER_CLASS = 2

# id order = sorted folder names
CLASS_DIRS = ["exserohilum_rostratum", "healthy", "sclerospora_graminicola"]
LABELS = {0: "Exserohilum leaf spot", 1: "Healthy", 2: "Downy mildew"}
CLASS_SCIENTIFIC = {
    "exserohilum_rostratum": "Exserohilum rostratum",
    "healthy": "none",
    "sclerospora_graminicola": "Sclerospora graminicola",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_dataset() -> pd.DataFrame:
    if not DEV_IMAGES.exists():
        print(f"Dev images not found at {DEV_IMAGES}; using {CSV_PATH.name}")
        return pd.read_csv(CSV_PATH)
    rows = []
    for cid, cname in enumerate(CLASS_DIRS):
        for p in sorted((DEV_IMAGES / cname).rglob("*")):
            if p.is_file() and p.suffix.lower() in SUFFIXES:
                rows.append({"image": p.name, "class_dir": cname, "label_id": cid,
                             **dict(zip(feature_names(), colour_features(p).tolist())),
                             "_path": str(p)})
    df = pd.DataFrame(rows)
    PROC.mkdir(parents=True, exist_ok=True)
    df.drop(columns="_path").to_csv(CSV_PATH, index=False)
    # small demo set: first N images (sorted) per class
    DEMO.mkdir(parents=True, exist_ok=True)
    for cname, g in df.groupby("class_dir"):
        for _, r in g.head(DEMO_PER_CLASS).iterrows():
            src = Path(r["_path"])
            shutil.copy2(src, DEMO / f"{cname}__{src.name}")
    return df.drop(columns="_path")


def main() -> None:
    ART.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    data = build_dataset()
    feats = feature_names()
    X, y = data[feats].astype(np.float32), data["label_id"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=200, min_samples_leaf=3, class_weight="balanced",
            random_state=SEED, n_jobs=1)),
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, pred)
    p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="macro", zero_division=0)
    cv_f1 = cross_val_score(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                            scoring="f1_macro", n_jobs=1)
    names = [LABELS[i] for i in sorted(LABELS)]
    report = classification_report(y_test, pred, target_names=names, digits=3, zero_division=0)

    fig, ax = plt.subplots(figsize=(6.2, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_test, pred, labels=sorted(LABELS)),
                           display_labels=names).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Pilot Module 6 pearl millet (test, optimistic split)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(RES / "confusion_matrix.png", dpi=140)
    plt.close(fig)

    onx = convert_sklearn(
        pipe, initial_types=[("input", FloatTensorType([None, len(feats)]))],
        target_opset=12, options={id(pipe.named_steps["model"]): {"zipmap": False}})
    onnx_path = ART / "disease_v1.onnx"
    onnx_path.write_bytes(onx.SerializeToString())

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_pred = np.asarray(sess.run(None, {sess.get_inputs()[0].name: X_test.to_numpy(np.float32)})[0]).reshape(-1).astype(int)
    agree = float(np.mean(ort_pred == pred))

    metrics = {
        "accuracy": round(float(acc), 4), "macro_precision": round(float(p), 4),
        "macro_recall": round(float(r), 4), "macro_f1": round(float(f1), 4),
        "cv5_macro_f1_mean": round(float(cv_f1.mean()), 4),
        "cv5_macro_f1_std": round(float(cv_f1.std()), 4),
        "onnx_sklearn_agreement": round(agree, 6),
    }
    counts = data["label_id"].value_counts().sort_index()
    schema = {
        "module": MODULE_ID, "version": VERSION, "task": "multiclass_classification",
        "framework": "onnxruntime",
        "android": {
            "runtime": "onnxruntime-android",
            "note": "ONNX takes 30 colour features, not pixels. The app must compute the same features first.",
            "input_tensor": f"float32[1,{len(feats)}]",
            "model_file": "disease_v1.onnx",
        },
        "feature_order": feats,
        "labels": [{"id": k, "name": v, "class_dir": CLASS_DIRS[k]} for k, v in LABELS.items()],
        "preprocessing": {
            "image": "resize 64x64 RGB, /255, per-channel mean, std, 8-bin histogram (normalised)",
            "model_bundled_in_onnx": ["median imputer", "standard scaler", "random forest"],
            "image_features_bundled_in_onnx": False,
        },
        "train": {
            "seed": SEED, "algorithm": "RandomForestClassifier", "n_estimators": 200,
            "min_samples_leaf": 3, "class_weight": "balanced",
            "n_rows_total": int(len(data)), "n_train": int(len(X_train)), "n_test": int(len(X_test)),
            "class_counts": {LABELS[int(k)]: int(v) for k, v in counts.items()},
            "split": "random stratified image split (no plant IDs; optimistic)",
            "missing_classes": ["pearl millet smut", "pyricularia (blast)"],
        },
        "metrics_holdout": metrics,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (ART / "schema.json").write_text(json.dumps(schema, indent=2))
    (ART / "labels.json").write_text(json.dumps(LABELS, indent=2))
    (ART / "manifest.json").write_text(json.dumps({
        "module": MODULE_ID, "version": VERSION,
        "disease_v1.onnx": sha256(onnx_path),
        "schema.json": sha256(ART / "schema.json"),
        "labels.json": sha256(ART / "labels.json"),
        "dataset": str(CSV_PATH.relative_to(ROOT)),
        "dataset_rows": int(len(data)),
    }, indent=2))
    (RES / "metrics.json").write_text(json.dumps(
        {"holdout": metrics, "classification_report": report}, indent=2))
    (RES / "classification_report.txt").write_text(report)
    print(report)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
