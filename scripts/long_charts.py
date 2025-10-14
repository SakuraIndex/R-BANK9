#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
桜Index: intraday変化率を算出（PNGは触らない）
- intraday.csv を元に {key}_stats.json と {key}_post_intraday.txt を更新
- 異常倍率（ゼロ割）や極小始値を自動除外して安定化
"""

import os, json
from datetime import datetime, time as dtime
from pathlib import Path
import pandas as pd
import numpy as np

OUTDIR = Path("docs/outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

INDEX_KEY = os.environ.get("INDEX_KEY", "index").lower()
MARKET_TZ = os.environ.get("MARKET_TZ", "Asia/Tokyo")
SESSION_START = os.environ.get("SESSION_START", "09:00")
SESSION_END   = os.environ.get("SESSION_END", "15:30")

# 指数ごとのスケール固定
SCALE_MAP = {
    "scoin_plus": "price",
    "rbank9": "pct",
    "ain10": "pct",
    "astra4": "fraction",
}


def _read_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    df = pd.read_csv(path)
    time_cols = [c for c in df.columns if c.lower() in ("datetime", "timestamp", "time", "日時", "date")]
    if not time_cols:
        raise ValueError(f"{path} に日時列がありません")
    df["ts_utc"] = pd.to_datetime(df[time_cols[0]], utc=True, errors="coerce")
    val_col = [c for c in df.columns if c.lower() not in ("datetime", "timestamp", "time", "日時", "date")][-1]
    df["value"] = pd.to_numeric(df[val_col], errors="coerce")
    return df.dropna(subset=["ts_utc", "value"])


def _first_last_valid(values):
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return None, None

    abs_mean = np.nanmean(np.abs(arr))
    threshold = abs_mean / 100 if abs_mean != 0 else 1e-6
    filtered = arr[np.abs(arr) > threshold]

    if len(filtered) < 2:
        filtered = arr

    start = filtered[0]
    end = filtered[-1]

    # 明らかに異常（倍率100倍超）なら補正
    if start != 0 and abs(end / start) > 100:
        print(f"[warn] Extreme ratio detected ({end/start:.1f}x) — clamping start.")
        start = np.median(filtered)

    return start, end


def _compute_change(vals, scale):
    s, e = _first_last_valid(vals)
    if s is None or e is None:
        return 0.0

    if scale == "price":
        pct = (e / s - 1) * 100
    elif scale == "pct":
        pct = e - s
    else:
        pct = (e - s) * 100
    return round(pct, 6)


def main():
    csv_path = OUTDIR / f"{INDEX_KEY}_intraday.csv"
    df = _read_csv(csv_path)

    scale = SCALE_MAP.get(INDEX_KEY, "pct")
    pct_1d = _compute_change(df["value"].tolist(), scale)

    stats = {
        "index_key": INDEX_KEY,
        "pct_1d": pct_1d,
        "scale": scale,
        "updated_at": datetime.utcnow().isoformat() + "Z"
    }

    json_path = OUTDIR / f"{INDEX_KEY}_stats.json"
    json_path.write_text(json.dumps(stats, ensure_ascii=False), encoding="utf-8")

    txt_path = OUTDIR / f"{INDEX_KEY}_post_intraday.txt"
    sign = "+" if pct_1d >= 0 else ""
    txt_path.write_text(f"{INDEX_KEY.upper()} 1d: {sign}{pct_1d:.2f}%", encoding="utf-8")

    print(f"[{INDEX_KEY}] change={pct_1d:.3f}% (scale={scale})")


if __name__ == "__main__":
    main()
