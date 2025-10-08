#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R-BANK9: intraday から日次の終値を作り、docs/outputs/rbank9_history.csv を更新する。
- intraday: docs/outputs/rbank9_intraday.csv
- history : docs/outputs/rbank9_history.csv （無ければ作成）
- 1日1行（date, value, volume=0）をベースにユニーク日付でマージして保存
"""

from __future__ import annotations

import os
import re
from typing import Optional

import pandas as pd

OUTPUT_DIR = "docs/outputs"
INDEX_KEY = "rbank9"  # 固定

INTRADAY_CSV = os.path.join(OUTPUT_DIR, f"{INDEX_KEY}_intraday.csv")
HISTORY_CSV   = os.path.join(OUTPUT_DIR, f"{INDEX_KEY}_history.csv")

RAW_TZ = "Asia/Tokyo"     # intraday の想定タイムゾーン
DISPLAY_TZ = "Asia/Tokyo" # 出力も JST

def log(msg: str):
    print(f"[update_history] {msg}")

def parse_time_any(x, raw_tz: str, display_tz: str):
    if pd.isna(x):
        return pd.NaT
    s = str(x).strip()

    # UNIX秒
    if re.fullmatch(r"\d{10}", s):
        return pd.Timestamp(int(s), unit="s", tz="UTC").tz_convert(display_tz)

    # pandas に委譲
    t = pd.to_datetime(s, errors="coerce", utc=False)
    if pd.isna(t):
        return pd.NaT

    # tz 付与/変換
    if t.tzinfo is None:
        t = t.tz_localize(raw_tz)
    return t.tz_convert(display_tz)

def pick_time_col(cols_lower: list[str]) -> Optional[str]:
    for c in ["datetime", "time", "timestamp", "date"]:
        if c in cols_lower:
            return c
    # あいまい一致
    hits = [i for i, c in enumerate(cols_lower) if ("time" in c) or ("date" in c)]
    return cols_lower[hits[0]] if hits else None

def pick_value_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        lc = str(c).lower()
        if any(k in lc for k in ["value", "close", "index", "price", "終値"]):
            return c
    # それでも無ければ最初の数値列
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None

def read_intraday(path: str) -> pd.DataFrame:
    """intraday CSV を読み取り、time/value（と任意 volume）へ正規化。"""
    if not os.path.exists(path):
        log(f"not found: {path}")
        return pd.DataFrame(columns=["time", "value", "volume"])

    df = pd.read_csv(path)
    raw_cols = list(df.columns)
    cols_lower = [str(c).strip().lower() for c in raw_cols]
    df.columns = cols_lower

    # コメント列 (#...) は除外
    df = df[[c for c in df.columns if not str(c).strip().startswith("#")]]

    # 時刻列
    tcol = pick_time_col(cols_lower)
    if tcol is None or tcol not in df.columns:
        raise KeyError(f"No time-like column. columns={list(df.columns)}")

    # 値列
    vcol = pick_value_col(df)
    if vcol is None or vcol not in df.columns:
        # 板形式（銘柄ごと複数列）の場合は数値列平均
        num_cols = []
        for c in df.columns:
            if c == tcol:
                continue
            as_num = pd.to_numeric(df[c], errors="coerce")
            if as_num.notna().sum() > 0:
                num_cols.append(c)
        if not num_cols:
            return pd.DataFrame(columns=["time", "value", "volume"])
        value = df[num_cols].apply(lambda s: pd.to_numeric(s, errors="coerce")).mean(axis=1)
    else:
        value = pd.to_numeric(df[vcol], errors="coerce")

    out = pd.DataFrame()
    out["time"] = df[tcol].apply(lambda x: parse_time_any(x, RAW_TZ, DISPLAY_TZ))
    out["value"] = value
    out["volume"] = 0

    out = out.dropna(subset=["time", "value"]).sort_values("time").reset_index(drop=True)
    return out

def to_daily_last(intraday: pd.DataFrame) -> pd.DataFrame:
    """JST の日付ごとに終値（最後の値）を抽出。"""
    if intraday.empty:
        return intraday
    d = intraday.copy()
    d["date"] = d["time"].dt.tz_convert(DISPLAY_TZ).dt.date
    g = d.groupby("date", as_index=False).agg({"value": "last", "volume": "sum"})
    # date を ISO 文字列へ
    g["date"] = pd.to_datetime(g["date"]).dt.strftime("%Y-%m-%d")
    g = g[["date", "value", "volume"]]
    return g

def load_history(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["date", "value", "volume"])
    df = pd.read_csv(path)
    # 列名ゆらぎ吸収
    lc = [c.lower() for c in df.columns]
    rename = {}
    if "date" not in lc:
        # 旧 "time" 保存など
        if "time" in lc:
            rename[df.columns[lc.index("time")]] = "date"
    if "value" not in lc:
        # 旧 "close" など
        for k in ["close", "index", "price"]:
            if k in lc:
                rename[df.columns[lc.index(k)]] = "value"
                break
    df = df.rename(columns=rename)
    if "volume" not in [c.lower() for c in df.columns]:
        df["volume"] = 0
    # date 正規化
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["date", "value"]).drop_duplicates(subset=["date"], keep="last")
    df = df.sort_values("date").reset_index(drop=True)
    return df[["date", "value", "volume"]]

def save_history(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def main():
    intraday = read_intraday(INTRADAY_CSV)
    if intraday.empty:
        log("intraday empty -> skip")
        return

    daily_new = to_daily_last(intraday)
    if daily_new.empty:
        log("daily_new empty -> skip")
        return

    hist_old = load_history(HISTORY_CSV)

    # 同一 date は新データを優先
    merged = pd.concat([hist_old[~hist_old["date"].isin(daily_new["date"])], daily_new], ignore_index=True)
    merged = merged.sort_values("date").reset_index(drop=True)

    # 直近 2 年分くらいに抑える（任意）
    if len(merged) > 0:
        cutoff = pd.Timestamp.today(tz=DISPLAY_TZ) - pd.Timedelta(days=730)
        merged = merged[pd.to_datetime(merged["date"]) >= cutoff.tz_localize(None)]

    save_history(merged, HISTORY_CSV)
    log(f"updated history rows={len(merged)} -> {HISTORY_CSV}")

if __name__ == "__main__":
    main()
