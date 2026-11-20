"""Rebuild Module 1 pilot dataset, model, ONNX artifact, and metrics."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
import onnxruntime as ort

from .parse_shc import FEATURES, parse_ahmedpur

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC, ART, RES = ROOT / "data/raw", ROOT / "data/processed", ROOT / "artifacts", ROOT / "results"
LABELS = {0: "Less Fertile", 1: "Fertile", 2: "Highly Fertile"}
SEED = 42


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_dataset() -> pd.DataFrame:
    shc = parse_ahmedpur(RAW / "india_shc_ahmedpur_maharashtra.csv")
    lab = pd.read_csv(RAW / "soil_fertility_labeled.csv").drop_duplicates(subset=FEATURES).copy()
    lab["source"] = "public_soil_fertility_labeled"
    lab["sample_id"] = [f"LAB_{i:05d}" for i in range(len(lab))]

    cr = pd.read_csv(RAW / "india_crop_recommendation_npk.csv")
    cr_rows = []
    for i, row in cr.iterrows():
        n, p, k = float(row["N"]), float(row["P"]), float(row["K"])
        scores = [
            0 if n < 50 else (1 if n < 100 else 2),
            0 if p < 40 else (1 if p < 70 else 2),
            0 if k < 40 else (1 if k < 80 else 2),
        ]
        mean_sc = float(np.mean(scores))
        cr_rows.append({
            "sample_id": f"CR_{i}",
            "source": "india_crop_recommendation_npk_rule",
            "N": n, "P": p, "K": k, "ph": float(row["ph"]),
            "ec": np.nan, "oc": np.nan, "S": np.nan, "zn": np.nan,
            "fe": np.nan, "cu": np.nan, "Mn": np.nan, "B": np.nan,
            "fertility": 0 if mean_sc < 0.75 else (1 if mean_sc < 1.5 else 2),
        })
    cr_df = pd.DataFrame(cr_rows)
    cr_df, _ = train_test_split(
        cr_df, train_size=min(600, len(cr_df)), stratify=cr_df["fertility"], random_state=SEED
    )

    data = pd.concat(
        [
            lab[FEATURES + ["fertility", "source", "sample_id"]],
            shc[FEATURES + ["fertility", "source", "sample_id"]],
            cr_df[FEATURES + ["fertility", "source", "sample_id"]],
        ],
        ignore_index=True,
    )
    data = data[(data["ph"].isna()) | ((data["ph"] >= 3.5) & (data["ph"] <= 10.5))]
    data = data.drop_duplicates(subset=FEATURES + ["fertility"])
    PROC.mkdir(parents=True, exist_ok=True)
    data.to_csv(PROC / "soil_fertility_india_v1.csv", index=False)
    return data


def main() -> None:
    for p in (ART, RES):
        p.mkdir(parents=True, exist_ok=True)
    data = build_dataset()
    X, y = data[FEATURES], data["fertility"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=3,
            class_weight="balanced", random_state=SEED, n_jobs=1,
        )),
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, pred)
    p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="macro", zero_division=0)
    cv_f1 = cross_val_score(
        pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
        scoring="f1_macro", n_jobs=1,
    )
    report = classification_report(
        y_test, pred, target_names=[LABELS[i] for i in sorted(LABELS)], digits=3
    )

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, pred), display_labels=["Less", "Fertile", "High"]
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Pilot Module 1 soil fertility (test)")
    fig.tight_layout()
    fig.savefig(RES / "confusion_matrix.png", dpi=140)
    plt.close()

    onx = convert_sklearn(
        pipe,
        initial_types=[("input", FloatTensorType([None, len(FEATURES)]))],
        target_opset=12,
        options={id(pipe.named_steps["model"]): {"zipmap": False}},
    )
    onnx_path = ART / "soil_fertility_v1.onnx"
    onnx_path.write_bytes(onx.SerializeToString())

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_pred = sess.run(None, {sess.get_inputs()[0].name: X_test.to_numpy(dtype=np.float32)})[0]
    ort_pred = np.asarray(ort_pred).reshape(-1).astype(int)
    agree = float(np.mean(ort_pred == pred))

    schema = {
        "module": "module_01_soil_fertility",
        "version": "1.0.0",
        "task": "multiclass_classification",
        "framework": "onnxruntime",
        "android": {
            "runtime": "onnxruntime-android",
            "note": "Runs on-device. No network bridge or tunnel required for inference.",
            "input_tensor": "float32[1,12]",
            "model_file": "soil_fertility_v1.onnx",
        },
        "feature_order": FEATURES,
        "labels": [{"id": k, "name": v} for k, v in LABELS.items()],
        "preprocessing": {"bundled_in_onnx": True},
        "train": {
            "seed": SEED,
            "algorithm": "RandomForestClassifier",
            "n_rows_total": int(len(data)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "sources": sorted(data["source"].unique().tolist()),
        },
        "metrics_holdout": {
            "accuracy": round(float(acc), 4),
            "macro_precision": round(float(p), 4),
            "macro_recall": round(float(r), 4),
            "macro_f1": round(float(f1), 4),
            "cv5_macro_f1_mean": round(float(cv_f1.mean()), 4),
            "cv5_macro_f1_std": round(float(cv_f1.std()), 4),
            "onnx_sklearn_agreement": round(agree, 6),
        },
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (ART / "schema.json").write_text(json.dumps(schema, indent=2))
    (ART / "labels.json").write_text(json.dumps(LABELS, indent=2))
    (ART / "manifest.json").write_text(json.dumps({
        "soil_fertility_v1.onnx": sha256(onnx_path),
        "schema.json": sha256(ART / "schema.json"),
        "labels.json": sha256(ART / "labels.json"),
        "dataset_rows": int(len(data)),
    }, indent=2))
    (RES / "metrics.json").write_text(json.dumps({"holdout": schema["metrics_holdout"], "classification_report": report}, indent=2))
    (RES / "classification_report.txt").write_text(report)
    print(json.dumps(schema["metrics_holdout"], indent=2))


if __name__ == "__main__":
    main()
