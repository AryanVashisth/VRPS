"""Parse India Soil Health Card style CSV exports into numeric features."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["N", "P", "K", "ph", "ec", "oc", "S", "zn", "fe", "cu", "Mn", "B"]


def split_val_status(cell):
    if pd.isna(cell):
        return np.nan, None
    s = str(cell).strip()
    m = re.match(r"^([-+]?[0-9]*\.?[0-9]+)\s*([A-Za-z]*)", s)
    if not m:
        return np.nan, None
    val = float(m.group(1))
    st = (m.group(2) or "").strip().upper() or None
    if st in {"VL", "L", "D"}:
        return val, "L"
    if st == "M":
        return val, "M"
    if st in {"H", "VH"}:
        return val, "H"
    return val, st


def status_to_score(code):
    return {"L": 0, "M": 1, "H": 2}.get(code, np.nan)


def parse_ahmedpur(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None)
    colmap = {
        "ph": 2, "ec": 3, "oc": 5, "N": 7, "P": 8, "K": 9, "S": 10, "zn": 11,
        "fe": 12, "cu": 13, "Mn": 14, "B": 15,
    }
    rows = []
    for i in range(4, len(raw)):
        r = raw.iloc[i]
        sample = r.iloc[1]
        if pd.isna(sample) or "Block/" in str(sample) or "Village:" in str(sample):
            continue
        if not re.search(r"\d", str(r.iloc[0])) and not re.match(r"^MH", str(sample)):
            continue
        rec = {"sample_id": str(sample), "source": "india_shc_ahmedpur_latur_mh"}
        scores = []
        for feat, idx in colmap.items():
            if idx >= len(r):
                rec[feat] = np.nan
                continue
            val, code = split_val_status(r.iloc[idx])
            rec[feat] = val
            if feat in {"N", "P", "K", "oc"} and code in {"L", "M", "H"}:
                scores.append(status_to_score(code))
        if len(scores) < 3:
            continue
        mean_sc = float(np.mean(scores))
        rec["fertility"] = 0 if mean_sc < 0.75 else (1 if mean_sc < 1.5 else 2)
        rows.append(rec)
    return pd.DataFrame(rows)
