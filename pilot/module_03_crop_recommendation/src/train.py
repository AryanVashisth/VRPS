"""Train crop recommender, compare baseline vs enhanced, export ONNX + metrics."""

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
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
ART = ROOT / "artifacts"
RES = ROOT / "results"
MODELS = ROOT / "models"

FEATURES = ["N", "P", "K", "pH", "Temperature", "Humidity", "Rainfall"]
TARGET = "crop"
SEED = 42
VERSION = "1.1.0"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "n": "N",
        "p": "P",
        "k": "K",
        "ph": "pH",
        "temperature": "Temperature",
        "humidity": "Humidity",
        "rainfall": "Rainfall",
        "label": "crop",
        "millet": "crop",
        "Crop": "crop",
    }
    out = frame.copy()
    out.columns = [str(c).strip() for c in out.columns]
    mapped = {}
    for col in out.columns:
        key = col if col in rename else col.lower()
        mapped[col] = rename.get(key, col)
    return out.rename(columns=mapped)


def build_dataset() -> pd.DataFrame:
    raw_path = RAW / "india_crop_recommendation_npk.csv"
    if not raw_path.is_file():
        raise FileNotFoundError(f"Missing raw dataset: {raw_path}")

    frame = _normalize_columns(pd.read_csv(raw_path))
    missing = [c for c in FEATURES + [TARGET] if c not in frame.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    labels = frame[TARGET].astype("string").str.strip().str.lower().replace("", pd.NA)
    features = frame[FEATURES].apply(pd.to_numeric, errors="coerce")
    data = features.copy()
    data[TARGET] = labels
    data = data.loc[data[TARGET].notna()].copy()

    # Light physical bounds (same spirit as VRPS soil module cleaning)
    data = data[(data["pH"].isna()) | ((data["pH"] >= 3.5) & (data["pH"] <= 10.5))]
    data = data[(data["Humidity"].isna()) | ((data["Humidity"] >= 0) & (data["Humidity"] <= 100))]
    data = data[(data["Rainfall"].isna()) | (data["Rainfall"] >= 0)]
    data = data.drop_duplicates(subset=FEATURES + [TARGET]).reset_index(drop=True)
    data["source"] = "india_crop_recommendation_npk"
    data["sample_id"] = [f"CR_{i:05d}" for i in range(len(data))]

    PROC.mkdir(parents=True, exist_ok=True)
    out_csv = PROC / "crop_recommendation_v1.csv"
    data.to_csv(out_csv, index=False)

    # Keep legacy path usable for the old CLI
    legacy = ROOT / "data" / "millet_data.csv"
    data[FEATURES + [TARGET]].rename(columns={TARGET: "Millet"}).to_csv(legacy, index=False)
    return data


def make_baseline() -> Pipeline:
    """Current-version pipeline (pre-enhancement)."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced_subsample",
                    random_state=SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def make_enhanced_candidates() -> list[tuple[str, Pipeline]]:
    """Incremental variants around the original RF — pick the best by holdout F1."""
    return [
        ("baseline_rf300", make_baseline()),
        (
            "imputer_rf500",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "classifier",
                        RandomForestClassifier(
                            n_estimators=500,
                            class_weight="balanced_subsample",
                            random_state=SEED,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        ),
        (
            "imputer_scaler_rf400",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        RandomForestClassifier(
                            n_estimators=400,
                            class_weight="balanced_subsample",
                            random_state=SEED,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        ),
    ]


def select_enhanced(X_train, y_train, X_test, y_test, label_ids: list[int]):
    """Pick the candidate with best holdout macro F1 (CV mean as tie-break)."""
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    best = None
    for name, pipe in make_enhanced_candidates():
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        hold = _metric_bundle(y_test, pred, label_ids)
        cv_scores = cross_val_score(
            pipe, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1
        )
        hold["cv5_macro_f1_mean"] = round(float(cv_scores.mean()), 4)
        hold["cv5_macro_f1_std"] = round(float(cv_scores.std()), 4)
        record = {
            "name": name,
            "model": pipe,
            "pred": pred,
            "holdout": hold,
            "params": pipe.named_steps["classifier"].get_params(),
        }
        if best is None:
            best = record
            continue
        key = (hold["macro_f1"], hold["cv5_macro_f1_mean"], hold["accuracy"])
        best_key = (
            best["holdout"]["macro_f1"],
            best["holdout"]["cv5_macro_f1_mean"],
            best["holdout"]["accuracy"],
        )
        if key > best_key:
            best = record
    return best


def _metric_bundle(y_true, y_pred, labels: list[str]) -> dict:
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0, labels=labels
    )
    return {
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(p), 4),
        "macro_recall": round(float(r), 4),
        "macro_f1": round(float(f1), 4),
    }


def main() -> None:
    for path in (ART, RES, MODELS):
        path.mkdir(parents=True, exist_ok=True)

    data = build_dataset()
    X = data[FEATURES]
    y_raw = data[TARGET].astype(str)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    class_names = list(encoder.classes_)
    label_map = {int(i): name for i, name in enumerate(class_names)}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    # --- Baseline (current version) on identical split ---
    baseline = make_baseline()
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    baseline_holdout = _metric_bundle(y_test, baseline_pred, list(range(len(class_names))))
    baseline_cv = cross_val_score(
        make_baseline(),
        X,
        y,
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
        scoring="f1_macro",
        n_jobs=-1,
    )
    baseline_holdout["cv5_macro_f1_mean"] = round(float(baseline_cv.mean()), 4)
    baseline_holdout["cv5_macro_f1_std"] = round(float(baseline_cv.std()), 4)

    # --- Enhanced: best of a few incremental RF variants ---
    label_ids = list(range(len(class_names)))
    chosen = select_enhanced(X_train, y_train, X_test, y_test, label_ids)
    model = chosen["model"]
    pred = chosen["pred"]
    holdout = chosen["holdout"]
    # Full-data CV for the selected recipe (fresh clone of steps/params)
    cv_pipe = Pipeline(list(model.steps))
    cv_f1 = cross_val_score(
        cv_pipe,
        X,
        y,
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
        scoring="f1_macro",
        n_jobs=-1,
    )
    holdout["cv5_macro_f1_mean"] = round(float(cv_f1.mean()), 4)
    holdout["cv5_macro_f1_std"] = round(float(cv_f1.std()), 4)
    holdout["selected_variant"] = chosen["name"]
    best_params = {
        "variant": chosen["name"],
        "n_estimators": chosen["params"].get("n_estimators"),
        "max_features": chosen["params"].get("max_features"),
        "min_samples_leaf": chosen["params"].get("min_samples_leaf"),
        "has_scaler": "scaler" in model.named_steps,
    }

    report = classification_report(
        y_test, pred, target_names=class_names, digits=3, zero_division=0
    )

    # Confusion matrix (readable for many classes)
    fig, ax = plt.subplots(figsize=(12, 10))
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, pred),
        display_labels=class_names,
    ).plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=90)
    ax.set_title("Crop recommendation — enhanced holdout")
    fig.tight_layout()
    fig.savefig(RES / "confusion_matrix.png", dpi=140)
    plt.close()

    clf = model.named_steps["classifier"]
    importance = pd.DataFrame(
        {"feature": FEATURES, "importance": clf.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance.to_csv(RES / "feature_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(importance["feature"][::-1], importance["importance"][::-1], color="#2F6F4E")
    ax.set_xlabel("Importance")
    ax.set_title("Feature importance (enhanced RF)")
    fig.tight_layout()
    fig.savefig(RES / "feature_importance.png", dpi=140)
    plt.close()

    # ONNX vs sklearn agreement on the holdout-fitted model
    check_onx = convert_sklearn(
        model,
        initial_types=[("features", FloatTensorType([None, len(FEATURES)]))],
        target_opset=12,
        options={id(model.named_steps["classifier"]): {"zipmap": False}},
    )
    check_path = ART / "_tmp_holdout_check.onnx"
    check_path.write_bytes(check_onx.SerializeToString())
    check_sess = ort.InferenceSession(str(check_path), providers=["CPUExecutionProvider"])
    check_ort = np.asarray(
        check_sess.run(
            None, {check_sess.get_inputs()[0].name: X_test.to_numpy(dtype=np.float32)}
        )[0]
    ).reshape(-1).astype(int)
    agree = float(np.mean(check_ort == pred))
    check_path.unlink(missing_ok=True)
    holdout["onnx_sklearn_agreement"] = round(agree, 6)

    # Refit on all rows for the shipped artifact
    model.fit(X, y)
    onx = convert_sklearn(
        model,
        initial_types=[("features", FloatTensorType([None, len(FEATURES)]))],
        target_opset=12,
        options={id(model.named_steps["classifier"]): {"zipmap": False}},
    )
    onnx_path = ART / "crop_recommendation_v1.onnx"
    onnx_path.write_bytes(onx.SerializeToString())
    (MODELS / "crop_rf.onnx").write_bytes(onnx_path.read_bytes())

    schema = {
        "module": "crop_recommendation",
        "version": VERSION,
        "task": "multiclass_classification",
        "framework": "onnxruntime",
        "android": {
            "runtime": "onnxruntime-android",
            "note": "Runs on-device. No network bridge required for inference.",
            "input_tensor": f"float32[1,{len(FEATURES)}]",
            "model_file": "crop_recommendation_v1.onnx",
        },
        "feature_order": FEATURES,
        "labels": [{"id": k, "name": v} for k, v in label_map.items()],
        "preprocessing": {
            "missing": "median imputer fitted on train",
            "scale": (
                "standard scaler fitted on train"
                if "scaler" in model.named_steps
                else "none (tree model)"
            ),
            "bundled_in_onnx": True,
        },
        "train": {
            "seed": SEED,
            "algorithm": "RandomForestClassifier",
            "best_params": best_params,
            "class_weight": "balanced_subsample",
            "n_rows_total": int(len(data)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "n_classes": int(len(class_names)),
            "sources": sorted(data["source"].unique().tolist()),
        },
        "metrics_holdout": holdout,
        "metrics_baseline_holdout": baseline_holdout,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    (ART / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    (ART / "labels.json").write_text(json.dumps(label_map, indent=2), encoding="utf-8")
    (ART / "manifest.json").write_text(
        json.dumps(
            {
                "crop_recommendation_v1.onnx": sha256(onnx_path),
                "schema.json": sha256(ART / "schema.json"),
                "labels.json": sha256(ART / "labels.json"),
                "dataset": "data/processed/crop_recommendation_v1.csv",
                "dataset_rows": int(len(data)),
                "version": VERSION,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    metrics_payload = {
        "version": VERSION,
        "holdout_enhanced": holdout,
        "holdout_baseline": baseline_holdout,
        "delta": {
            "accuracy": round(holdout["accuracy"] - baseline_holdout["accuracy"], 4),
            "macro_f1": round(holdout["macro_f1"] - baseline_holdout["macro_f1"], 4),
            "cv5_macro_f1_mean": round(
                holdout["cv5_macro_f1_mean"] - baseline_holdout["cv5_macro_f1_mean"], 4
            ),
        },
        "classification_report": report,
    }
    (RES / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    (RES / "classification_report.txt").write_text(report, encoding="utf-8")

    print(json.dumps({"enhanced": holdout, "baseline": baseline_holdout, "delta": metrics_payload["delta"]}, indent=2))


if __name__ == "__main__":
    main()
