#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
当日(Asia/Tokyo)の終値レベルを取得し、docs/outputs/rbank9_history.csv に upsert する。
優先ソース:
  1) docs/outputs/rbank9_stats.json の {level / index / close / value}
  2) scripts/level_from_intraday.py で intraday.csv から推定（保険）
"""

import os, json
from pathlib import Path
import pandas as pd

INDEX_KEY = os.environ.get("INDEX_KEY", "rbank9")
OUT_DIR   = Path(os.environ.get("OUT_DIR", "docs/outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

HIST  = OUT_DIR / f"{INDEX_KEY}_history.csv"
STATS = OUT_DIR / f"{INDEX_KEY}_stats.json"

def _level_from_stats() -> float | None:
    if not STATS.exists():
        return None
    try:
        j = json.loads(STATS.read_text())
    except Exception:
        return None
    for k in ("level","index","close","close_level","value","val"):
        v = j.get(k)
        if isinstance(v, (int,float)):
            return float(v)
    return None

def _level_from_intraday_fallback() -> float | None:
    # intraday.csv を読んで最終行の値を終値レベルと見なす保険導線
    try:
        from level_from_intraday import guess_close_level
        return guess_close_level(INDEX_KEY, OUT_DIR)
    except Exception:
        return None

def _load_hist() -> pd.DataFrame:
    if HIST.exists():
        try:
            df = pd.read_csv(HIST)
        except Exception:
            df = pd.DataFrame(columns=["date","value"])
    else:
        df = pd.DataFrame(columns=["date","value"])
    # 列正規化
    cols = {c.lower(): c for c in df.columns}
    tcol = cols.get("date")
    vcol = cols.get("value")
    if tcol is None or vcol is None:
        df = pd.DataFrame(columns=["date","value"])
    return df

def main():
    today = pd.Timestamp.now(tz="Asia/Tokyo").date()
    level = _level_from_stats()
    if level is None:
        level = _level_from_intraday_fallback()
    if level is None:
        print("[append_history] no level -> skip")
        return

    df = _load_hist()
    # upsert
    mask = (df["date"].astype(str) == str(today))
    if mask.any():
        df.loc[mask, "value"] = level
    else:
        df = pd.concat([df, pd.DataFrame([{"date": str(today), "value": level}])],
                       ignore_index=True)

    df = df.dropna().sort_values("date")
    df.to_csv(HIST, index=False)
    print(f"[append_history] upsert {today} -> {level}")

if __name__ == "__main__":
    main()
