#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
R-BANK9: intraday から日次終値を計算して docs/outputs/rbank9_history.csv に追記/更新
- 板形式（銘柄ごとに列）/ 通常形式（time/value[/volume]）の両対応
- 当日セッション(JST 9:00-15:30)の最後の値=終値として保存
"""

from __future__ import annotations
import os
import re
import pandas as pd
from typing import Optional, List, Tuple

INDEX_KEY = os.environ.get("INDEX_KEY", "").lower() or "rbank9"
OUT_DIR = "docs/outputs"
INTRADAY_CSV = f"{OUT_DIR}/{INDEX_KEY}_intraday.csv"
HISTORY_CSV   = f"{OUT_DIR}/{INDEX_KEY}_history.csv"

DISPLAY_TZ = "Asia/Tokyo"
SESSION_TZ = "Asia/Tokyo"
SESSION_START = (9, 0)
SESSION_END   = (15, 30)

def log(msg: str):
    print(f"[update_history] {msg}")

def pick_time_col(cols_lower: List[str]) -> Optional[str]:
    for k in ["time", "timestamp", "datetime", "date"]:
        if k in cols_lower:
            return k
    # Unnamed: 0 を time として扱う（インデックス書き出しの救済）
    if "unnamed: 0" in cols_lower:
        return "unnamed: 0"
    any_time = [c for c in cols_lower if ("time" in c) or ("date" in c)]
    return any_time[0] if any_time else None

def parse_time_any(x, raw_tz: str, display_tz: str):
    if pd.isna(x):
        return pd.NaT
    s = str(x).strip()
    # 10桁UNIX seconds
    if re.fullmatch(r"\d{10}", s):
        return pd.Timestamp(int(s), unit="s", tz="UTC").tz_convert(display_tz)
    # 13桁UNIX millis
    if re.fullmatch(r"\d{13}", s):
        return pd.Timestamp(int(s), unit="ms", tz="UTC").tz_convert(display_tz)
    t = pd.to_datetime(s, errors="coerce")
    if pd.isna(t):
        return pd.NaT
    if t.tzinfo is None:
        t = t.tz_localize(raw_tz)
    return t.tz_convert(display_tz)

def read_intraday(path: str, raw_tz: str, display_tz: str) -> pd.DataFrame:
    """
    - time/value[/volume] 形式 or 板形式に対応
    - 返り値: ["time","value","volume"]
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=["time", "value", "volume"])

    df = pd.read_csv(path)
    # コメント列(# ...)は除外
    df = df[[c for c in df.columns if not str(c).strip().startswith("#")]]

    raw_cols = list(df.columns)
    cols_lower = [str(c).strip().lower() for c in raw_cols]
    df.columns = cols_lower

    tcol = pick_time_col(cols_lower)
    if tcol is None:
        raise KeyError(f"No time-like column found. columns={list(df.columns)}")

    # 値列の推定
    # 1) 明示列 value/price/index/score など
    vcol = None
    for k in ["value", "price", "index", "score", "終値"]:
        if k in cols_lower:
            vcol = k
            break

    out = pd.DataFrame()
    out["time"] = df[tcol].apply(lambda x: parse_time_any(x, raw_tz, display_tz))

    if vcol is not None:
        out["value"] = pd.to_numeric(df[vcol], errors="coerce")
    else:
        # 2) 板形式: time 以外で数値化できる列の平均
        num_cols = []
        for c in df.columns:
            if c == tcol:
                continue
            as_num = pd.to_numeric(df[c], errors="coerce")
            if as_num.notna().sum() > 0:
                num_cols.append(c)
        if not num_cols:
            return pd.DataFrame(columns=["time", "value", "volume"])
        vals_df = df[num_cols].apply(lambda s: pd.to_numeric(s, errors="coerce"))
        out["value"] = vals_df.mean(axis=1)

    out["volume"] = 0
    out = out.dropna(subset=["time", "value"]).sort_values("time").reset_index(drop=True)
    return out

def session_frame_for(ts_jst: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    j = ts_jst.tz_convert(SESSION_TZ)
    d = j.date()
    start = pd.Timestamp(d.year, d.month, d.day, SESSION_START[0], SESSION_START[1], tz=SESSION_TZ).tz_convert(DISPLAY_TZ)
    end   = pd.Timestamp(d.year, d.month, d.day, SESSION_END[0],   SESSION_END[1],   tz=SESSION_TZ).tz_convert(DISPLAY_TZ)
    return start, end

def main():
    if not INDEX_KEY:
        raise SystemExit("ERROR: INDEX_KEY not set")

    intraday = read_intraday(INTRADAY_CSV, DISPLAY_TZ, DISPLAY_TZ)
    if intraday.empty:
        log("intraday empty -> skip")
        return

    last_ts = intraday["time"].max()
    start, end = session_frame_for(last_ts)
    day_slice = intraday[(intraday["time"] >= start) & (intraday["time"] <= end)].copy()
    if day_slice.empty:
        log("intraday within session empty -> skip")
        return

    close_val = day_slice["value"].iloc[-1]
    day = end.tz_convert(DISPLAY_TZ).date()

    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(HISTORY_CSV):
        hist = pd.read_csv(HISTORY_CSV)
        if not {"date","value"}.issubset(set(hist.columns)):
            # 既存が壊れていたら作り直し
            hist = pd.DataFrame(columns=["date","value"])
    else:
        hist = pd.DataFrame(columns=["date","value"])

    if (hist["date"] == str(day)).any():
        hist.loc[hist["date"] == str(day), "value"] = float(close_val)
        log(f"updated history for {day}: {close_val}")
    else:
        hist = pd.concat([hist, pd.DataFrame([{"date": str(day), "value": float(close_val)}])], ignore_index=True)
        log(f"appended history for {day}: {close_val}")

    hist = hist.sort_values("date").reset_index(drop=True)
    hist.to_csv(HISTORY_CSV, index=False)
    log(f"saved: {HISTORY_CSV} rows={len(hist)}")

if __name__ == "__main__":
    main()
