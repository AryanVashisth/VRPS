"""Train Module 4 fertilizer recommender, export ONNX, write metrics and plots."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd
import sklearn
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/fertilizer_prediction.csv"
ART, RES = ROOT / "artifacts", ROOT / "results"
MODULE = "module_04_fertilizer_recommendation"
MODEL_FILE = "model.onnx"
SEED = 42

NUMERIC = ["Temperature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous"]
CATEGORICAL = ["Soil_Type", "Crop_Type"]  # shown as "Soil Type" / "Crop Type" in schema
TARGET = "Fertilizer Name"
ONNX_NAME = {c: c for c in NUMERIC + CATEGORICAL}
DISPLAY = {c: c.replace("_", " ") for c in NUMERIC + CATEGORICAL}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_clean() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(RAW)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Temparature": "Temperature", "Soil Type": "Soil_Type", "Crop Type": "Crop_Type"})
    for c in CATEGORICAL + [TARGET]:
        df[c] = df[c].astype(str).str.strip()
    df[TARGET] = df[TARGET].replace({"10/26/2026": "10-26-26"})
    rows_in = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    return df, {"rows_in_file": rows_in, "rows_after_dedupe": len(df)}


def make_pipeline() -> Pipeline:
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    model = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=2, class_weight="balanced", random_state=SEED, n_jobs=1,
    )
    return Pipeline([("prep", prep), ("model", model)])


def save_confusion(y_true, y_pred, names, path: Path, title: str) -> None:
    m = confusion_matrix(y_true, y_pred, labels=range(len(names)))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, int(m[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def scores(y, p) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y, p)), 4),
        "macro_precision": round(float(precision_score(y, p, average="macro", zero_division=0)), 4),
        "macro_recall": round(float(recall_score(y, p, average="macro", zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y, p, average="macro", zero_division=0)), 4),
    }


def to_onnx_frame(X: pd.DataFrame, session: ort.InferenceSession) -> dict:
    feed = {}
    for inp in session.get_inputs():
        col = next(c for c, n in ONNX_NAME.items() if n == inp.name)
        if col in NUMERIC:
            feed[inp.name] = X[col].to_numpy(dtype=np.float32).reshape(-1, 1)
        else:
            feed[inp.name] = X[col].to_numpy(dtype=object).reshape(-1, 1)
    return feed


def main() -> None:
    ART.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    df, notes = load_clean()
    names = sorted(df[TARGET].unique())
    label_id = {n: i for i, n in enumerate(names)}
    X = df[NUMERIC + CATEGORICAL]
    y = df[TARGET].map(label_id).astype(int)

    # Reported estimate: stratified 5-fold CV (n is small).
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    cvs = cross_validate(
        make_pipeline(), X, y, cv=cv,
        scoring={"accuracy": "accuracy", "macro_f1": "f1_macro"},
    )
    oof = cross_val_predict(make_pipeline(), X, y, cv=cv)
    cv_metrics = {
        "accuracy_mean": round(float(cvs["test_accuracy"].mean()), 4),
        "accuracy_std": round(float(cvs["test_accuracy"].std()), 4),
        "macro_f1_mean": round(float(cvs["test_macro_f1"].mean()), 4),
        "macro_f1_std": round(float(cvs["test_macro_f1"].std()), 4),
        "oof": scores(y, oof),
    }
    save_confusion(y, oof, names, RES / "confusion_matrix_cv.png", "Out-of-fold (5-fold) fertilizer model")

    # Secondary: single stratified holdout (about 20 rows; very noisy).
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    hold = make_pipeline().fit(X_tr, y_tr)
    hold_pred = hold.predict(X_te)
    hold_metrics = {"n_train": len(X_tr), "n_test": len(X_te), **scores(y_te, hold_pred)}
    save_confusion(y_te, hold_pred, names, RES / "confusion_matrix_holdout.png",
                   f"Holdout (n={len(X_te)}) fertilizer model")
    report = classification_report(y, oof, labels=range(len(names)), target_names=names,
                                   zero_division=0, digits=3)

    # Ship model: fit on all cleaned rows.
    final = make_pipeline().fit(X, y)
    initial = [
        (ONNX_NAME[c], FloatTensorType([None, 1])) for c in NUMERIC
    ] + [
        (ONNX_NAME[c], StringTensorType([None, 1])) for c in CATEGORICAL
    ]
    onx = convert_sklearn(
        final, initial_types=initial, target_opset={"": 15, "ai.onnx.ml": 3},
        options={id(final.named_steps["model"]): {"zipmap": False}},
    )
    onnx_path = ART / MODEL_FILE
    onnx_path.write_bytes(onx.SerializeToString())

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = sess.run(None, to_onnx_frame(X, sess))
    ort_pred = np.asarray(out[0]).reshape(-1).astype(int)
    ort_proba = np.asarray(out[1])
    sk_pred = final.predict(X)
    agreement = float(np.mean(ort_pred == sk_pred))
    max_proba_diff = float(np.abs(ort_proba - final.predict_proba(X)).max())

    schema = {
        "module": MODULE,
        "version": "1.0.0",
        "task": "multiclass_classification",
        "framework": "onnxruntime",
        "android": {
            "runtime": "onnxruntime-android",
            "note": "Runs on-device. No network required for inference.",
            "model_file": MODEL_FILE,
        },
        "inputs": (
            [{"name": DISPLAY[c], "onnx_input": ONNX_NAME[c], "dtype": "float32", "shape": [1, 1], "unit": None}
             for c in NUMERIC]
            + [{"name": DISPLAY[c], "onnx_input": ONNX_NAME[c], "dtype": "string", "shape": [1, 1],
                "allowed": sorted(df[c].unique().tolist())} for c in CATEGORICAL]
        ),
        "outputs": ["label (int64[1])", "probabilities (float32[1,7], index = label id)"],
        "preprocessing": {
            "bundled_in_onnx": True,
            "numeric": "median imputer",
            "categorical": "OneHotEncoder(handle_unknown=ignore)",
        },
        "labels": [{"id": i, "name": n} for i, n in enumerate(names)],
        "train": {
            "seed": SEED,
            "algorithm": "RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight=balanced)",
            "n_rows": int(len(df)),
            "class_counts": {n: int((df[TARGET] == n).sum()) for n in names},
            "cleaning": ["exact duplicates dropped", "label 10/26/2026 -> 10-26-26", "Temparature -> Temperature"],
        },
        "metrics": {
            "cv5_macro_f1_mean": cv_metrics["macro_f1_mean"],
            "cv5_macro_f1_std": cv_metrics["macro_f1_std"],
            "holdout_macro_f1": hold_metrics["macro_f1"],
            "onnx_sklearn_agreement": round(agreement, 6),
        },
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (ART / "schema.json").write_text(json.dumps(schema, indent=2))
    (ART / "labels.json").write_text(json.dumps({str(i): n for i, n in enumerate(names)}, indent=2))
    (ART / "manifest.json").write_text(json.dumps({
        MODEL_FILE: sha256(onnx_path),
        "schema.json": sha256(ART / "schema.json"),
        "labels.json": sha256(ART / "labels.json"),
        "dataset": "data/raw/fertilizer_prediction.csv",
        "dataset_rows_after_dedupe": int(len(df)),
        "sklearn": sklearn.__version__,
    }, indent=2))
    (RES / "metrics.json").write_text(json.dumps({
        "cleaning": notes,
        "cv5": cv_metrics,
        "holdout": hold_metrics,
        "onnx_check_all_rows": {"label_agreement": round(agreement, 6), "max_abs_proba_diff": max_proba_diff},
        "classification_report_oof": report,
    }, indent=2))
    print(json.dumps({"cv5": cv_metrics, "holdout": hold_metrics,
                      "onnx_agreement": agreement, "max_proba_diff": max_proba_diff}, indent=2))


if __name__ == "__main__":
    main()
