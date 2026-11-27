"""VRPS pilot console: local FastAPI front for all ONNX pilot modules.

Run from this folder:
  uvicorn app:app --reload --port 8765
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

PILOT_ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"

MODULE_DIRS = {
    "module_01_soil_fertility": PILOT_ROOT / "module_01_soil_fertility",
    "module_02_soil_moisture": PILOT_ROOT / "module_02_soil_moisture",
    "module_03_crop_recommendation": PILOT_ROOT / "module_03_crop_recommendation",
    "module_04_fertilizer_recommendation": PILOT_ROOT / "module_04_fertilizer_recommendation",
    "module_05_crop_yield": PILOT_ROOT / "module_05_crop_yield",
    "module_06_pearl_millet_disease": PILOT_ROOT / "module_06_pearl_millet_disease",
}

DISPLAY = {
    "module_01_soil_fertility": {
        "short": "Soil fertility",
        "original": "Original #1",
        "task": "classification",
    },
    "module_02_soil_moisture": {
        "short": "Soil moisture (4h)",
        "original": "Original #2",
        "task": "regression",
    },
    "module_03_crop_recommendation": {
        "short": "Crop recommendation",
        "original": "Original #6",
        "task": "classification",
    },
    "module_04_fertilizer_recommendation": {
        "short": "Fertilizer",
        "original": "Original #4",
        "task": "classification",
    },
    "module_05_crop_yield": {
        "short": "Crop yield",
        "original": "Original #5",
        "task": "regression",
    },
    "module_06_pearl_millet_disease": {
        "short": "Pearl millet disease",
        "original": "Original #3",
        "task": "classification",
    },
}

app = FastAPI(title="VRPS Pilot Console", version="0.1.0")
_loaders: dict[str, Any] = {}


def _load_schema(module_id: str) -> dict | None:
    path = MODULE_DIRS[module_id] / "artifacts" / "schema.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _ensure_path(module_id: str) -> Path:
    root = MODULE_DIRS.get(module_id)
    if root is None or not root.exists():
        raise HTTPException(404, f"Unknown module: {module_id}")
    return root


def _import_infer(module_id: str):
    if module_id in _loaders:
        return _loaders[module_id]
    root = _ensure_path(module_id)
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    # Clear sibling module name collisions by importing via path files
    import importlib.util

    infer_path = src / "infer.py"
    if not infer_path.exists():
        raise HTTPException(503, f"{module_id} has no infer.py yet")
    spec = importlib.util.spec_from_file_location(f"vrps_{module_id}_infer", infer_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    _loaders[module_id] = mod
    return mod


class InferBody(BaseModel):
    features: dict[str, Any] = Field(default_factory=dict)


@app.get("/api/health")
def health() -> dict:
    ready = []
    missing = []
    for mid in MODULE_DIRS:
        schema = _load_schema(mid)
        onnx_ok = any((MODULE_DIRS[mid] / "artifacts").glob("*.onnx")) if MODULE_DIRS[mid].exists() else False
        if schema and onnx_ok:
            ready.append(mid)
        else:
            missing.append(mid)
    return {"status": "ok", "ready": ready, "pending": missing}


def _feature_order(schema: dict | None) -> list[str]:
    if not schema:
        return []
    # Prefer human form fields (yield encodes cats outside ONNX).
    nums = list(schema.get("numeric_features") or [])
    cats = list(schema.get("categorical_features") or [])
    if nums or cats:
        return nums + cats
    if schema.get("inputs"):
        return [item["name"] for item in schema["inputs"]]
    return list(schema.get("feature_order") or [])


@app.get("/api/modules")
def list_modules() -> list[dict]:
    out = []
    for mid, meta in DISPLAY.items():
        schema = _load_schema(mid)
        metrics = None
        mpath = MODULE_DIRS[mid] / "results" / "metrics.json"
        if mpath.exists():
            metrics = json.loads(mpath.read_text())
        onnx_ok = any((MODULE_DIRS[mid] / "artifacts").glob("*.onnx")) if MODULE_DIRS[mid].exists() else False
        out.append(
            {
                "id": mid,
                "short": meta["short"],
                "original": meta["original"],
                "task": meta["task"],
                "ready": bool(schema) and onnx_ok,
                "version": (schema or {}).get("version"),
                "feature_order": _feature_order(schema),
                "labels": (schema or {}).get("labels"),
                "schema": schema,
                "metrics": metrics,
                "sample": _sample_for(mid, schema),
            }
        )
    return out


def _sample_for(module_id: str, schema: dict | None) -> dict:
    if not schema:
        return {}
    samples = {
        "module_01_soil_fertility": {
            "N": 90, "P": 42, "K": 43, "ph": 6.5, "ec": 0.5, "oc": 0.6,
            "S": 12, "zn": 0.6, "fe": 5.0, "cu": 0.3, "Mn": 3.0, "B": 0.5,
        },
        "module_02_soil_moisture": {
            "lag_0": 0.22, "lag_1": 0.22, "lag_4": 0.21, "lag_16": 0.20, "lag_96": 0.19,
            "mean_last_4h": 0.215, "std_last_4h": 0.004, "delta_1": 0.0, "delta_4": 0.01,
        },
        "module_03_crop_recommendation": {
            "N": 90, "P": 42, "K": 43, "pH": 6.5, "Temperature": 25.0,
            "Humidity": 80.0, "Rainfall": 200.0,
        },
        "module_04_fertilizer_recommendation": {
            "Temperature": 26, "Humidity": 52, "Moisture": 38,
            "Nitrogen": 37, "Potassium": 0, "Phosphorous": 0,
            "Soil Type": "Sandy", "Crop Type": "Maize",
        },
        "module_05_crop_yield": {
            "Crop_Year": 2018, "Area": 1200.0, "Annual_Rainfall": 850.0,
            "Crop": "Bajra", "Season": "Kharif", "State": "Rajasthan",
        },
        "module_06_pearl_millet_disease": {},
    }
    return samples.get(module_id, {})


@app.post("/api/infer/{module_id}")
def infer(module_id: str, body: InferBody) -> dict:
    _ensure_path(module_id)
    mod = _import_infer(module_id)
    try:
        if module_id == "module_01_soil_fertility":
            model = mod.SoilFertilityModel()
            return {"ok": True, "result": model.predict_one(body.features)}
        if module_id == "module_02_soil_moisture":
            # Prefer class name from module
            Model = getattr(mod, "SoilMoistureModel", None) or getattr(mod, "MoistureModel", None)
            if Model is None:
                # fall back to any callable predict_one at module level
                if hasattr(mod, "predict_one"):
                    return {"ok": True, "result": mod.predict_one(body.features)}
                raise AttributeError("No moisture model class")
            model = Model()
            return {"ok": True, "result": model.predict_one(body.features)}
        if module_id == "module_03_crop_recommendation":
            Model = getattr(mod, "CropRecommendationModel", None) or getattr(mod, "CropModel", None)
            if Model is None and hasattr(mod, "predict_one"):
                return {"ok": True, "result": mod.predict_one(body.features)}
            model = Model()
            return {"ok": True, "result": model.predict_one(body.features)}
        if module_id == "module_04_fertilizer_recommendation":
            Model = getattr(mod, "FertilizerModel", None) or getattr(mod, "FertilizerRecommendationModel", None)
            model = Model()
            return {"ok": True, "result": model.predict_one(body.features)}
        if module_id == "module_05_crop_yield":
            Model = getattr(mod, "YieldModel", None) or getattr(mod, "CropYieldModel", None)
            model = Model()
            return {"ok": True, "result": model.predict_one(body.features)}
        if module_id == "module_06_pearl_millet_disease":
            Model = getattr(mod, "DiseaseModel", None) or getattr(mod, "PearlMilletDiseaseModel", None)
            model = Model()
            if body.features:
                if hasattr(model, "predict_from_features"):
                    return {"ok": True, "result": model.predict_from_features(body.features)}
                return {"ok": True, "result": model.predict_one(body.features)}
            raise HTTPException(400, "Disease module expects an image upload or feature vector")
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Inference failed: {exc}") from exc
    raise HTTPException(404, module_id)


@app.post("/api/infer/{module_id}/image")
async def infer_image(module_id: str, file: UploadFile = File(...)) -> dict:
    if module_id != "module_06_pearl_millet_disease":
        raise HTTPException(400, "Image upload is only for disease module")
    _ensure_path(module_id)
    mod = _import_infer(module_id)
    Model = getattr(mod, "DiseaseModel", None) or getattr(mod, "PearlMilletDiseaseModel", None)
    model = Model()
    suffix = Path(file.filename or "leaf.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        path = tmp.name
    try:
        result = model.predict_from_image(path)
        return {"ok": True, "result": result}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Image inference failed: {exc}") from exc
    finally:
        Path(path).unlink(missing_ok=True)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
