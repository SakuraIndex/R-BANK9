#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
intraday.csv から「終値レベル」を推定する保険スクリプト。
- CSV: docs/outputs/rbank9_intraday.csv
- 列名は柔軟に解釈: ts/time/date(datetime系) と val/value/index/level(numeric系)
- 最終時刻の値を返す
"""

from pathlib import Path
import pandas as pd

def _best_datetime_col(df: pd.DataFrame) -> str | None:
    hints = ["ts","time","timestamp","date","datetime"]
    low = [c.lower().strip() for c in df.columns]
    for h in hints:
        if h in low:
            c = df.columns[low.index(h)]
            s = pd.to_datetime(df[c], errors="coerce", utc=False)
            if s.notna().sum() >= max(3, len(s)//5):
                df[c] = s
                return c
    # 最大ヒット列
    best, ok = None, -1
    for c in df.columns:
        s = pd.to_datetime(df[c], errors="coerce", utc=False)
        if s.notna().sum() > ok:
            ok, best = s.notna().sum(), c
            df[c] = s
    return best

def _best_value_col(df: pd.DataFrame, tcol: str) -> str | None:
    cand = []
    for c in df.columns:
        if c == tcol: continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= max(3, len(s)//5):
            cand.append((c, s.notna().sum()))
            df[c] = s
    cand.sort(key=lambda x: x[1], reverse=True)
    return cand[0][0] if cand else None

def guess_close_level(index_key: str, out_dir: Path) -> float | None:
    csv = out_dir / f"{index_key}_intraday.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    if df.shape[1] < 2:
        return None
    tcol = _best_datetime_col(df.copy())
    if not tcol: return None
    vcol = _best_value_col(df.copy(), tcol)
    if not vcol: return None

    d = pd.DataFrame({
        "ts":  pd.to_datetime(df[tcol], errors="coerce", utc=False),
        "val": pd.to_numeric(df[vcol], errors="coerce")
    }).dropna().sort_values("ts")
    if d.empty:
        return None
    # 当日分のみの最終（なければ全体の最終）
    today = pd.Timestamp.now(tz="Asia/Tokyo").date()
    dd = d[d["ts"].dt.date == today]
    if dd.empty: dd = d
    return float(dd.iloc[-1]["val"])
