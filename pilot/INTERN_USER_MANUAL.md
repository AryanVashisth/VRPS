# VRPS Pilot Intern Manual

How to open the code, run a model, and avoid overclaiming results.

Last updated: 27 November 2026

---

## 1. What this project is

VRPS is building toward a **mobile advisory app** for farmers (India, millet-aware).  
Right now you are looking at **pilot modules**: small, offline-ready models that prove the phone path.

| You have | You do not have yet |
| --- | --- |
| Six ONNX model folders | A finished Android app |
| A laptop Pilot Console | Field-proven accuracy |
| Schemas for inputs/outputs | Sensor drivers on device |

If a mentor asks “does this advise farmers today?”, the correct answer is:  
**No. It proves offline inference and data pipelines. Field validation comes next.**

---

## 2. Folder map (start here)

```
pilot/
  README.md                 # module table
  VALIDATION_AUDIT.md       # honesty / India-fit audit
  INTERN_USER_MANUAL.md     # this file
  dashboard/                # local web console to test all modules
  module_01_soil_fertility/
  module_02_soil_moisture/
  module_03_crop_recommendation/
  module_04_fertilizer_recommendation/
  module_05_crop_yield/
  module_06_pearl_millet_disease/
```

GitHub (branch `pilot`): https://github.com/AryanVashisth/VRPS/tree/pilot/pilot

### Original module numbers vs pilot folders

The research plan numbered Disease as #3 and Crop as #6.  
Pilot shipping order put Crop earlier as `module_03`. Map:

| Pilot folder | Original plan |
| --- | --- |
| module_01_soil_fertility | #1 Soil |
| module_02_soil_moisture | #2 Moisture |
| module_03_crop_recommendation | #6 Crop |
| module_04_fertilizer_recommendation | #4 Fertilizer |
| module_05_crop_yield | #5 Yield |
| module_06_pearl_millet_disease | #3 Disease |

---

## 3. Anatomy of one module

Every module follows the same shape:

```
module_XX_.../
  artifacts/
    *.onnx          # model the phone will load
    schema.json     # feature names, version, Android notes
    labels.json     # id → class name (classification only)
  src/
    train.py        # rebuild model (heavier)
    infer.py        # run prediction (what the app mirrors)
  data/             # raw / processed tables (or demo images)
  results/          # metrics.json and plots
  docs/             # short pilot notes
  README.md
  requirements.txt
```

**Golden rule:** Android and Python must use the same `feature_order` (or `inputs`) from `schema.json`. Never invent column names.

---

## 4. One-time setup

```bash
cd "pilot"
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r module_01_soil_fertility/requirements.txt
pip install -r dashboard/requirements.txt
```

Use the shared `pilot/.venv` for demos. Do not create a `.venv` inside each module folder.

---

## 5. Run a single prediction (CLI)

Example — soil fertility:

```bash
cd module_01_soil_fertility
../.venv/bin/python -m src.infer
```

Example — disease from an image:

```bash
cd module_06_pearl_millet_disease
../.venv/bin/python -m src.infer
# or pass a path:
../.venv/bin/python -m src.infer data/demo_images/healthy__PESGL_1.jpg
```

Each `infer.py` prints JSON: label (or number), confidence when available, `model_version`.

---

## 6. Run the Pilot Console (all modules)

```bash
cd dashboard
../.venv/bin/uvicorn app:app --port 8765
```

Open http://127.0.0.1:8765

1. Start on **Stack overview** (six modules, ready/pending)  
2. Open one module  
3. **Load sample** → **Run inference**  
4. For disease, upload a leaf image from `module_06_.../data/demo_images/`

Predictions stay on your laptop (ONNX Runtime). No cloud call.

---

## 7. What each module does (intern-level)

### Module 1 — Soil fertility
- **In:** N, P, K, ph, ec, oc, S, zn, fe, cu, Mn, B  
- **Out:** Less Fertile / Fertile / Highly Fertile  
- **Metric to quote:** macro F1 (not only accuracy)  
- **Careful:** training mixes India SHC rows with other tables; units are not identical across sources. Ask before treating phone inputs as kg/ha.

### Module 2 — Soil moisture (4 hours ahead)
- **In:** lag and rolling features from a moisture time series  
- **Out:** predicted water content; `recommended` follows persistence while that policy wins  
- **Careful:** dataset is not yet campus India. Quote the policy, not “our Indian field forecast”.

### Module 3 — Crop recommendation
- **In:** N, P, K, pH, Temperature, Humidity, Rainfall  
- **Out:** one of 22 crop names  
- **Careful:** **no pearl millet / bajra class**. Never demo this as millet advice.

### Module 4 — Fertilizer name
- **In:** weather + moisture + N/P/K + soil type + crop type  
- **Out:** fertilizer product name (Urea, DAP, grades, …)  
- **Careful:** only 99 clean rows. This is not a Soil Health Card dose (kg/ha) engine.

### Module 5 — Crop yield
- **In:** year, area, rainfall, crop, season, state  
- **Out:** yield number from the table’s units  
- **Careful:** coconut rows explode all-crop error. Prefer Bajra / non-coconut metrics. Rajasthan is missing from this file.

### Module 6 — Pearl millet leaf disease
- **In:** leaf image → 30 colour features → ONNX  
- **Out:** Healthy / Downy mildew / Exserohilum leaf spot  
- **Careful:** small image set; missing disease archives; holdout F1 of 1.0 is not trustworthy. Use CV numbers and say “lab baseline”.

---

## 8. How to read `metrics.json` without fooling yourself

1. Prefer **macro F1** for classification (every class counts).  
2. Prefer **MAE** for regression; also check a crop slice (e.g. Bajra).  
3. If train and test can share the same leaf or the same duplicated row, scores look too good.  
4. `onnx_sklearn_agreement` only proves conversion, not field truth.  
5. If confidence is ~0.4 to 0.5, say “uncertain” in any demo.

---

## 9. Meeting arc (so you know why folders exist)

| Meets | Theme |
| --- | --- |
| 30 to 33 | Sensors → schema → campus Dataset v0.1 → live Module 1 trial |
| 34 to 37 | Remaining pilots + console → close stack → Month 6 integration writing |

Do not redo sensor workshops when the topic is fertilizer/yield/disease.

---

## 10. Coding habits for this repo

1. Change features? Update `schema.json`, retrain, re-export ONNX, re-check `infer.py`.  
2. Never commit `.venv`, `__pycache__`, or huge raw image archives.  
3. Do not add “Co-authored-by” or tool names on slides or commits.  
4. Keep README honesty lines when you touch a module.  
5. For Android later: one screen → one schema → one ONNX file → one result string.

---

## 11. Mini checklist before you demo

- [ ] Console shows the module as **ready**  
- [ ] You can explain inputs in one sentence  
- [ ] You can name the metric you trust  
- [ ] You say one honesty line (millet gap / small n / persistence / lab images)  
- [ ] You do not promise farmer accuracy

---

## 12. Who to ask

If schema fields and campus sensors disagree, stop and write the mismatch in the Sensor Schema Map (Meet 31 deliverable). Do not “fix” the model to match a wrong unit.
