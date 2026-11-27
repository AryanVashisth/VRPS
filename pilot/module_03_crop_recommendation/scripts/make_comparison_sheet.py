"""Build a one-page Current vs Enhanced comparison workbook."""

from __future__ import annotations

import json
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "Current_vs_Enhanced_Comparison.xlsx"
METRICS = json.loads((ROOT / "results" / "metrics.json").read_text(encoding="utf-8"))
SCHEMA = json.loads((ROOT / "artifacts" / "schema.json").read_text(encoding="utf-8"))


def style_header(cell, fill: str) -> None:
    cell.font = Font(name="Arial", bold=True, size=11, color="FFFFFF")
    cell.fill = PatternFill("solid", fgColor=fill)
    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)


def style_body(cell, bold: bool = False) -> None:
    cell.font = Font(name="Arial", size=10, bold=bold)
    cell.alignment = Alignment(vertical="center", wrap_text=True)


def main() -> None:
    enhanced = METRICS["holdout_enhanced"]
    baseline = METRICS["holdout_baseline"]
    n_rows = SCHEMA["train"]["n_rows_total"]
    n_classes = SCHEMA["train"]["n_classes"]
    variant = enhanced.get("selected_variant", "n/a")

    wb = Workbook()
    ws = wb.active
    ws.title = "Current vs Enhanced"

    thin = Border(
        left=Side(style="thin", color="B0B0B0"),
        right=Side(style="thin", color="B0B0B0"),
        top=Side(style="thin", color="B0B0B0"),
        bottom=Side(style="thin", color="B0B0B0"),
    )

    ws.merge_cells("A1:C1")
    ws["A1"] = "Crop Recommendation — Current vs Enhanced (1-page)"
    ws["A1"].font = Font(name="Arial", bold=True, size=14, color="1F3D2B")
    ws["A1"].alignment = Alignment(horizontal="left", vertical="center")

    ws.merge_cells("A2:C2")
    ws["A2"] = (
        f"Enhanced v{SCHEMA['version']} | data rows={n_rows} | classes={n_classes} | "
        f"selected variant={variant} | seed=42 | source=india_crop_recommendation_npk"
    )
    ws["A2"].font = Font(name="Arial", size=9, italic=True, color="555555")

    headers = ["Area", "Current version (before)", "Enhanced version (now)"]
    for col, text in enumerate(headers, start=1):
        cell = ws.cell(3, col, text)
        style_header(cell, "2F6F4E")
        cell.border = thin

    comparisons = [
        (
            "Project layout",
            "Flat scripts: train_model.py, predict.py, convert_model.py; models/",
            "VRPS-style: src/, data/raw|processed/, artifacts/, results/, docs/",
        ),
        (
            "Training data",
            "data/millet_data.csv header only (0 labeled rows) — could not retrain",
            f"Loaded + cleaned India crop CSV → processed table ({n_rows} rows, {n_classes} crops)",
        ),
        (
            "Model pipeline",
            "Imputer + RF(300, balanced_subsample); no model selection",
            "Candidate RF variants compared; ships best by holdout macro F1",
        ),
        (
            "Evaluation",
            "Print-only accuracy/report; no saved metrics or plots",
            "Holdout + 5-fold CV, confusion matrix, feature importance, metrics.json",
        ),
        (
            "Artifacts for deploy",
            "Single ONNX file only (ZipMap probability dict)",
            "ONNX + schema.json + labels.json + manifest.json (Android-ready, zipmap=False)",
        ),
        (
            "Inference API",
            "CLI prompts only",
            "CLI + src.infer CropRecommendationModel (JSON: crop, confidence, top3)",
        ),
        (
            "Holdout accuracy",
            "Not reproducible (empty training CSV; old ONNX had no saved metrics)",
            f"{enhanced['accuracy']:.2%} on stratified 20% holdout",
        ),
        (
            "Holdout macro F1",
            "Not available",
            f"{enhanced['macro_f1']:.2%}",
        ),
        (
            "5-fold CV macro F1",
            "Not available",
            f"{enhanced['cv5_macro_f1_mean']:.2%} ± {enhanced['cv5_macro_f1_std']:.2%}",
        ),
        (
            "ONNX ↔ sklearn agreement",
            "Not measured",
            f"{enhanced['onnx_sklearn_agreement']:.1%} on holdout labels",
        ),
        (
            "Top features",
            "Not reported",
            "Rainfall, Humidity, K (see results/feature_importance.csv)",
        ),
        (
            "How to rebuild",
            "python train_model.py (failed on empty CSV)",
            "python -m src.train   then   python predict.py --n ...",
        ),
    ]

    for r, (area, current, enhanced_text) in enumerate(comparisons, start=4):
        ws.cell(r, 1, area)
        ws.cell(r, 2, current)
        ws.cell(r, 3, enhanced_text)
        for c in range(1, 4):
            cell = ws.cell(r, c)
            style_body(cell, bold=(c == 1))
            cell.border = thin
            if c == 1:
                cell.fill = PatternFill("solid", fgColor="E8F2EC")

    note_row = 4 + len(comparisons) + 1
    ws.merge_cells(f"A{note_row}:C{note_row}")
    ws[f"A{note_row}"] = (
        "Note: Accuracy was already near ceiling once real data is used "
        f"(baseline RF recipe also {baseline['accuracy']:.2%} / CV F1 "
        f"{baseline['cv5_macro_f1_mean']:.2%} on the same split). "
        "Selected ship variant = "
        f"{variant}. Main upgrade vs the attached project: real data + VRPS layout + "
        "saved eval artifacts + Android-ready ONNX package. Source: results/metrics.json."
    )
    ws[f"A{note_row}"].font = Font(name="Arial", size=9, color="333333")
    ws[f"A{note_row}"].alignment = Alignment(wrap_text=True, vertical="top")

    ws.column_dimensions["A"].width = 28
    ws.column_dimensions["B"].width = 52
    ws.column_dimensions["C"].width = 58
    ws.row_dimensions[1].height = 22
    ws.row_dimensions[2].height = 18
    ws.row_dimensions[3].height = 24
    for r in range(4, 4 + len(comparisons)):
        ws.row_dimensions[r].height = 36
    ws.row_dimensions[note_row].height = 48

    ws.print_title_rows = "1:3"
    ws.page_setup.orientation = "landscape"
    ws.page_setup.fitToPage = True
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 1
    ws.page_margins.left = 0.4
    ws.page_margins.right = 0.4
    ws.page_margins.top = 0.4
    ws.page_margins.bottom = 0.4

    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
