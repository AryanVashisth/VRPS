# VRPS development baselines

Enhancing Millet Crop Production Using AI and IoT.

This folder is the intern starting point for the six analytical modules. Each module has its own data, training script, requirements file, and results. The modules are trained separately. They are not merged into one model. A later decision step is what will combine their outputs, after each baseline is valid on its own.

The scores below were produced on 15 October 2026 with seed `42`. Package versions are pinned in `requirements.txt`.

## How to run

From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd module_01_soil_nutrient
python train_baselines.py
```

Repeat the last two lines in any module folder. `download_data.py` fetches the public file again if `data/raw` is empty. The tables used for these results are already in each module.

The install is pandas, scikit-learn, matplotlib, and Pillow for the leaf module. There is no PyTorch and no saved model file. Training uses one CPU core. The leaf module keeps three small image classes. The smut and Pyricularia archives are about 230 MB and 370 MB, so they are not included.

`data_cache/` is only a scratch download area. Scripts do not read it.

## What a valid baseline means

Use the assigned dataset. Fit imputation, scaling, and encoding on the training rows only. Compare models on the same split. Keep a simple baseline in the table. Report more than accuracy. Do not describe a score as millet-field performance unless the rows actually come from millet fields.

## Modules

| Folder | Task | Data used | Result to read first | Model carried forward |
| --- | --- | --- | --- | --- |
| `module_01_soil_nutrient` | Fertility class | 880 unique soil rows | Macro F1 **0.864** (random forest) | Random forest, after more unique Highly Fertile rows |
| `module_02_irrigation_soil_moisture` | 4-hour soil-water forecast | Gallipoli 15-minute sensors | Persistence MAE **0.00073** | None yet. Persistence still wins |
| `module_03_pearl_millet_disease` | Leaf disease class | 80 images, 3 small classes | Colour features only. See that README | Random forest on colour features. No neural net in this folder |
| `module_04_fertilizer_recommendation` | Fertilizer name | 99 unique recommendations | Macro F1 **0.967** ± 0.065 | Random forest. Do not tune on 99 rows |
| `module_05_crop_yield` | Yield, future years | Indian states, 1997–2020 | Non-coconut MAE **1.65** (random forest) | Random forest, after the yield units are split |
| `module_06_crop_recommendation` | Crop label | 2,200 augmented rows, 22 crops | Macro F1 **0.993** | Random forest. This file has no millet class |

Each module README has the full table, the plots to open, and the reasons a high score can still be the wrong thing to trust.

## Shared rules in the code

- Seed is 42.
- Preprocessing sits inside a scikit-learn pipeline, so it is fit on training data only.
- Classification tables include a majority-class baseline and macro F1.
- Regression tables include a simple baseline, MAE, RMSE, and R².
- Exact duplicate rows are removed before scoring when the file repeats them.
- Production is not an input to the yield model, because yield is approximately production divided by area.
