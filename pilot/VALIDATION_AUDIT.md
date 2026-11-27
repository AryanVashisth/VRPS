# VRPS Pilot Stack — End-to-End Validation Audit

Date: 27 November 2026  
Scope: testing and external cross-checks only (no new model work)  
Audience: RA / mentor before sharing with interns or presenting

## 1. Buffer rule (how to present)

Six-month tracker goal for this phase was baselines and development plans. The pilot stack is ahead of that. Use a deliberate buffer:

| Layer | Meaning |
| --- | --- |
| Built (keep in repo) | All 6 ONNX pilots + local Pilot Console |
| Present publicly | Prove the **mobile path**: schema → ONNX → offline infer. Show 2 to 3 modules live, with honesty lines |
| Do not claim | Farmer advice, field accuracy, millet crop recommendation, clinic-grade disease diagnosis |

Rule of thumb: keep about **20% more** engineering in the repo than you demo, and claim about **20% less** capability than the repo contains.

**Safe demo set for presentations:** Module 1 (soil) + Module 3 (crop, with millet gap said aloud) + Console overview.  
**Keep in buffer (show only if asked):** Module 2 policy note, Module 4 fertilizer, Module 5 yield, Module 6 disease.

## 2. Mobile end-user path (what must work later)

Target flow for the future Android app:

1. User or sensor provides values matching `artifacts/schema.json` feature names  
2. App builds `float32` (or string) tensors in that order  
3. `onnxruntime-android` runs the `.onnx` file offline  
4. App maps label id → farmer text via `labels.json` / schema  

What is already true today:

- Every pilot has `schema.json` + ONNX + Python `infer.py` mirror  
- Local console at `pilot/dashboard` exercises the same contracts  
- No network is required for prediction  

What is **not** ready for a store app:

- No Android UI, no sensor drivers, no offline packaging of all models  
- No shared input form that maps campus sensors to every schema  
- Disease needs colour-feature extraction before ONNX (not raw camera → ONNX)  
- Several datasets are not millet-first or not India-complete (see below)

## 3. Dataset and India-fit scorecard

| Module | Data | India climate / soil fit | Prediction quality (honest) | Useful for mobile demo? |
| --- | --- | --- | --- | --- |
| M1 Soil fertility | SHC Ahmedpur (MH) + public + rule-labeled crop NPK | Good India soil story; **units mixed across sources** (SHC N median ~95, public ~257, crop-rule ~37) | Holdout macro F1 ~0.98 recovers SHC-style labels; not independent field grades | Yes, with unit note |
| M2 Moisture 4h | Zenodo port time series | Not campus India yet | Persistence beats Ridge; ship policy is correct | Demo only; say “policy, not Indian field model” |
| M3 Crop | Kaggle-style 22 crops, 2200 rows | Indian-style NPK + climate features; **no bajra / millet** | Holdout macro F1 ~0.995; sanity checks mostly sensible | Yes, if millet gap is stated |
| M4 Fertilizer | 99 rows after dedupe | Small, looks rule/synthetic; not SHC dose tables | CV macro F1 ~0.95; matches Urea/DAP sanity cases | Intern demo only |
| M5 Yield | India district yield table 1997–2020 | Broad India crops; **no Rajasthan in file** (largest bajra state missing) | Bajra MAE better than coconut-dominated all-crop MAE | Planning demo only |
| M6 Disease | ~80 leaf images, 3 classes | Pearl millet pathogens (good topic fit); missing smut / Pyricularia archives | Holdout F1 1.0 is not trustworthy; CV ~0.88; colour features | Lab smoke test only |

## 4. External cross-checks (sample predictions)

### Module 1 — soil fertility

| Input sketch | Model | External check |
| --- | --- | --- |
| Very low N/P/K, low OC | Less Fertile (conf ~0.83) | Aligns with ICAR low bands (N low &lt;280 kg/ha style ratings; OC low &lt;0.5%) — direction OK |
| Mid NPK sample | Less Fertile (conf ~0.45) | Ambiguous; low confidence — do not present as a confident farmer call |
| High NPK / higher OC | Fertile (conf ~0.45), not Highly Fertile | Highly Fertile is rare in training (~252/5921). Do not expect “Highly Fertile” often |

**Risk:** Mixing SHC kg/ha-scale values with crop-table NPK ratios without a documented unit bridge will confuse Android form design. Schema must declare units before sensor mapping freezes.

### Module 3 — crop recommendation

| Scenario | Model | External check |
| --- | --- | --- |
| High rain + high humidity rice-like (N90 P42 K43, rain 200, RH 80) | **jute** 0.54 / rice 0.46 | NRRI: rice likes ~1000–2000 mm annual rain and humid tropics. Our table’s rice rainfall median is ~233 (dataset scale). Borderline rice/jute — say “top-2”, not a single crop |
| Low humidity chickpea-like | chickpea ~0.95 | Matches dry-pulse niche in the same literature family — good |
| Low rain + high humidity | muskmelon ~0.94 | Consistent with melon rows in this table — OK for demo |

**Hard gap:** No pearl millet / bajra class. For a millet project, this module must never be presented as millet advice.

### Module 4 — fertilizer

| Case | Model | External check |
| --- | --- | --- |
| Maize, sandy, N high, P/K ~0 | Urea ~0.67 | Maize in India is N-hungry; urea as N source is directionally right (ICAR split-urea practice). Dose and splits are **not** predicted |
| Sugarcane, low N, high P | DAP ~0.68 | DAP is a P (and N) source — matches the table’s own logic |
| Balanced 15/15/15 | 17-17-17 ~0.41 | Weak confidence — do not demo |

**Hard gap:** Real Indian advice uses Soil Health Card + crop dose (kg/ha), not a 99-row name classifier. Present as “label lookup pilot”, not recommendation engine.

### Module 5 — yield

| Case | Model | External check |
| --- | --- | --- |
| Bajra, Rajasthan, 2018 | ~1.83 | Rajasthan state yield ~1.2 t/ha recently; irrigated surveys higher. Ballpark-ish, but **Rajasthan is absent from the CSV**, so the state code is unknown (−1). Do not claim Rajasthan accuracy |
| Rice, West Bengal | ~2.27 | Plausible order of magnitude for many grain rows |
| Coconut, Kerala | thousands | Coconut rows dominate error; numbers are not comparable to cereal t/ha |

### Module 2 — moisture

Persistence is returned as `recommended` while Ridge exists in ONNX. Matches metrics (persistence MAE better). Correct engineering honesty for mobile: prefer the better baseline.

### Module 6 — disease

Demo images classify to the expected class names. Images are from the training pool → smoke test only. Colour histograms are not a substitute for a field CNN or expert diagnosis.

## 5. End-to-end console / infer smoke (executed 27 Nov 2026)

All six modules: ONNX loads, `infer.py` demos run, dashboard `/api/health` reported 6 ready / 0 pending. Fertilizer, yield, fertility, crop, moisture, and disease image upload paths returned structured JSON.

## 6. What to fix later (backlog only — not done in this audit)

1. Document and enforce **one unit system** for M1 features (prefer SHC kg/ha + %).  
2. Add millet rows or a separate millet-suitability module before any millet crop UI.  
3. Yield table: add Rajasthan (and check Yield units per crop).  
4. Fertilizer: replace or supplement with SHC-linked dose tables.  
5. Disease: plant-level split + fuller class set before Android camera claim.  
6. Android: one shared “FeaturePacket” builder from `schema.json`.

## 7. Verdict

| Question | Answer |
| --- | --- |
| Is the stack useful as a **mobile architecture pilot**? | Yes |
| Is it useful as a **farmer decision app today**? | No |
| India soil/climate relevance | Partial — strongest on M1 SHC story and M3 climate features; weakest on millet crop label, Rajasthan yield, fertilizer doses, campus moisture |
| Present with buffer? | Yes — demo path + honesty, keep weaker modules in the repo buffer |
