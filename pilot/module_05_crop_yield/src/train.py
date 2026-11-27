"""Train Module 5 crop-yield regressor, export ONNX, write metrics and plots."""

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
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score, root_mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/crop_yield.csv"
ART, RES = ROOT / "artifacts", ROOT / "results"
MODULE = "module_05_crop_yield"
MODEL_FILE = "model.onnx"
SEED = 42
TRAIN_LAST_YEAR = 2016

NUMERIC = ["Crop_Year", "Area", "Annual_Rainfall"]
CATEGORICAL = ["Crop", "Season", "State"]
FEATURES = NUMERIC + [f"{c}_code" for c in CATEGORICAL]  # ONNX input order, float32[1,6]
TARGET = "Yield"
MILLET = "Bajra"  # pearl millet in this table
UNKNOWN_CODE = -1


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_clean() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(RAW)
    rows_in = len(df)
    for c in CATEGORICAL:
        df[c] = df[c].astype(str).str.strip()
    for c in NUMERIC + [TARGET]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=NUMERIC + CATEGORICAL + [TARGET])
    df = df[df["Area"] > 0].reset_index(drop=True)
    return df, {"rows_in_file": rows_in, "rows_used": len(df)}


def encode(df: pd.DataFrame, maps: dict) -> np.ndarray:
    cols = [df[c].to_numpy(dtype=np.float32) for c in NUMERIC]
    cols += [df[c].map(maps[c]).fillna(UNKNOWN_CODE).to_numpy(dtype=np.float32) for c in CATEGORICAL]
    return np.column_stack(cols).astype(np.float32)


def reg_scores(y, p) -> dict:
    if len(y) == 0:
        return {"n": 0}
    return {
        "n": int(len(y)),
        "mae": round(float(mean_absolute_error(y, p)), 4),
        "rmse": round(float(root_mean_squared_error(y, p)), 4),
        "median_ae": round(float(median_absolute_error(y, p)), 4),
        "r2": round(float(r2_score(y, p)), 4) if len(y) > 1 else None,
    }


def main() -> None:
    ART.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    df, notes = load_clean()
    train = df[df["Crop_Year"] <= TRAIN_LAST_YEAR]
    test = df[df["Crop_Year"] > TRAIN_LAST_YEAR]

    # Category maps from train rows only.
    maps = {c: {v: i for i, v in enumerate(sorted(train[c].unique()))} for c in CATEGORICAL}
    X_tr, X_te = encode(train, maps), encode(test, maps)
    y_tr, y_te = train[TARGET].to_numpy(), test[TARGET].to_numpy()
    unseen_test_rows = int((X_te[:, 3:] == UNKNOWN_CODE).any(axis=1).sum())

    model = RandomForestRegressor(
        n_estimators=100, max_depth=16, min_samples_leaf=5, random_state=SEED, n_jobs=1,
    ).fit(X_tr, y_tr)
    pred = model.predict(X_te)

    # Reference only: per-crop mean of train yields (not shipped).
    crop_mean = train.groupby("Crop")[TARGET].mean()
    base_pred = test["Crop"].map(crop_mean).fillna(train[TARGET].mean()).to_numpy()

    masks = {
        "all_test_rows": np.ones(len(test), bool),
        "excluding_coconut": (test["Crop"] != "Coconut").to_numpy(),
        "bajra_only": (test["Crop"] == MILLET).to_numpy(),
    }
    metrics = {
        name: {"random_forest": reg_scores(y_te[m], pred[m]), "crop_mean_baseline": reg_scores(y_te[m], base_pred[m])}
        for name, m in masks.items()
    }
    coconut = (test["Crop"] == "Coconut").to_numpy()
    abs_err = np.abs(y_te - pred)
    coconut_share = float(abs_err[coconut].sum() / abs_err.sum()) if abs_err.sum() else 0.0

    # Plots: residual scatter (non-coconut zoom beside full) and histogram.
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].scatter(y_te, pred, s=6, alpha=0.4)
    ax[0].plot([0, y_te.max()], [0, y_te.max()], "r--", lw=1)
    ax[0].set_title("All test rows (coconut stretches the axis)")
    nm = masks["excluding_coconut"]
    ax[1].scatter(y_te[nm], pred[nm], s=6, alpha=0.4)
    lim = float(max(y_te[nm].max(), pred[nm].max()))
    ax[1].plot([0, lim], [0, lim], "r--", lw=1)
    ax[1].set_title("Excluding coconut")
    for a in ax:
        a.set_xlabel("Actual yield")
        a.set_ylabel("Predicted yield")
    fig.tight_layout()
    fig.savefig(RES / "residual_scatter.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist((pred - y_te)[nm], bins=60, color="#3b6d8f")
    ax.set_xlabel("Predicted - actual yield (excluding coconut)")
    ax.set_ylabel("Rows")
    ax.set_title(f"Residuals, test years {TRAIN_LAST_YEAR + 1}+")
    fig.tight_layout()
    fig.savefig(RES / "residual_histogram.png", dpi=120)
    plt.close(fig)

    # ONNX export of the train-only model; encoding stays outside the graph (maps in schema.json).
    onx = convert_sklearn(
        model, initial_types=[("input", FloatTensorType([None, len(FEATURES)]))],
        target_opset={"": 15, "ai.onnx.ml": 3},
    )
    onnx_path = ART / MODEL_FILE
    onnx_path.write_bytes(onx.SerializeToString())
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_pred = sess.run(None, {"input": X_te})[0].reshape(-1)
    max_diff = float(np.abs(ort_pred - pred).max())
    rel = float(np.max(np.abs(ort_pred - pred) / np.maximum(np.abs(pred), 1e-6)))

    schema = {
        "module": MODULE,
        "version": "1.0.0",
        "task": "regression",
        "target": {"name": "Yield", "note": "Recorded yield (Production / Area) in the source table's units"},
        "framework": "onnxruntime",
        "android": {
            "runtime": "onnxruntime-android",
            "note": "Runs on-device. No network required for inference.",
            "model_file": MODEL_FILE,
            "input_tensor": f"float32[1,{len(FEATURES)}]",
        },
        "feature_order": FEATURES,
        "numeric_features": NUMERIC,
        "categorical_features": CATEGORICAL,
        "categorical_encoding": {
            "type": "integer_codes_outside_onnx",
            "fit_on": f"Crop_Year <= {TRAIN_LAST_YEAR}",
            "unknown_code": UNKNOWN_CODE,
            "strip_whitespace": True,
            "maps": maps,
        },
        "preprocessing": {"bundled_in_onnx": False, "note": "Apply category maps from this file before calling ONNX; no scaling needed."},
        "excluded_features": ["Production (leaks Yield)", "Fertilizer", "Pesticide"],
        "train": {
            "seed": SEED,
            "algorithm": "RandomForestRegressor(n_estimators=100, max_depth=16, min_samples_leaf=5)",
            "split": f"train Crop_Year <= {TRAIN_LAST_YEAR}, test later",
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "trained_on": "train split only (shipped model has not seen test years)",
        },
        "metrics": {
            "mae_all": metrics["all_test_rows"]["random_forest"]["mae"],
            "mae_excluding_coconut": metrics["excluding_coconut"]["random_forest"]["mae"],
            "mae_bajra": metrics["bajra_only"]["random_forest"]["mae"],
        },
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (ART / "schema.json").write_text(json.dumps(schema, indent=2))
    (ART / "manifest.json").write_text(json.dumps({
        MODEL_FILE: sha256(onnx_path),
        "schema.json": sha256(ART / "schema.json"),
        "dataset": "data/raw/crop_yield.csv",
        "dataset_rows_used": int(len(df)),
        "model_bytes": onnx_path.stat().st_size,
        "sklearn": sklearn.__version__,
    }, indent=2))
    out = {
        "cleaning": notes,
        "split": {"train_rows": len(train), "test_rows": len(test),
                  "test_rows_by_year": {str(k): int(v) for k, v in test["Crop_Year"].value_counts().sort_index().items()},
                  "test_rows_with_unseen_category": unseen_test_rows},
        "test": metrics,
        "coconut_share_of_total_abs_error": round(coconut_share, 4),
        "onnx_vs_sklearn_on_test": {"max_abs_diff": max_diff, "max_rel_diff": rel},
        "feature_importance": {f: round(float(v), 4) for f, v in zip(FEATURES, model.feature_importances_)},
    }
    (RES / "metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("model bytes", onnx_path.stat().st_size)


if __name__ == "__main__":
    main()
