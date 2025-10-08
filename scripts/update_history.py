#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Append/refresh daily history from intraday for Sakura Index repos.
- 板形式/縦持ちの両CSVを自動判別
- 先頭列 Unnamed: 0 を時刻として扱う
- 先頭が '#' の列（銘柄リストの説明など）は自動除外
- 数値化できる列を等加重平均して 'value' を作成
- display_tz（JST）の日付で終値（last）を履歴に反映

出力: docs/outputs/<index_key>_history.csv
"""

from __future__ import annotations

import os
import re
import pandas as pd

OUTPUT_DIR = "docs/outputs"

def log(msg: str):
    print(f"[update_history] {msg}")

def get_key() -> str:
    key = os.environ.get("INDEX_KEY", "").lower()
    if not key:
        raise SystemExit("ERROR: INDEX_KEY not set")
    return key

def parse_time_any(x, raw_tz: str, display_tz: str):
    if pd.isna(x):
        return pd.NaT
    s = str(x).strip()
    # UNIX秒(10桁)
    if re.fullmatch(r"\d{10}", s):
        return pd.to_datetime(int(s), unit="s", utc=True).tz_convert(display_tz)
    # 一般
    t = pd.to_datetime(s, errors="coerce")
    if pd.isna(t):
        return pd.NaT
    if t.tzinfo is None:
        # 入力は raw_tz のローカル時刻とみなす
        t = t.tz_localize(raw_tz)
    return t.tz_convert(display_tz)

def pick_time_col(cols: list[str]) -> str | None:
    # 代表的な名前
    for name in ["datetime", "timestamp", "time", "date"]:
        if name in cols:
            return name
    # pandas の index 書き出し
    for name in cols:
        if name.startswith("unnamed: 0"):
            return name
    # あいまい一致
    fuzz = [c for c in cols if ("time" in c) or ("date" in c)]
    return fuzz[0] if fuzz else None

def read_intraday(path: str, raw_tz: str, display_tz: str) -> pd.DataFrame:
    """
    intraday CSV を読み、必ず [time, value] を返す。
    """
    if not path or not os.path.exists(path):
        log(f"no intraday file: {path}")
        return pd.DataFrame(columns=["time", "value"])

    df = pd.read_csv(path)
    # 列名を正規化（小文字/trim）
    raw_cols = list(df.columns)
    cols = [str(c).strip().lower() for c in raw_cols]
    df.columns = cols

    # コメント列/空列を除去
    drop_cols = [c for c in df.columns if c.startswith("#") or c == ""]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # 時刻列
    tcol = pick_time_col(list(df.columns))
    if tcol is None:
        raise KeyError(f"No time-like column found. columns={list(df.columns)}")

    # 値の決定: 'value' があればそれ、なければ数値化できる列の等加重平均
    num_candidates: list[str] = []
    if "value" in df.columns:
        num_candidates = ["value"]
    else:
        for c in df.columns:
            if c == tcol:
                continue
            # ティッカーやメモ列想定を数値化試行
            ser = pd.to_numeric(df[c], errors="coerce")
            if ser.notna().sum() > 0:
                num_candidates.append(c)

    if not num_candidates:
        # 何も数値化できなければ空
        return pd.DataFrame(columns=["time", "value"])

    out = pd.DataFrame()
    out["time"] = df[tcol].apply(lambda x: parse_time_any(x, raw_tz, display_tz))
    # 値
    vals = df[num_candidates].apply(lambda s: pd.to_numeric(s, errors="coerce"))
    out["value"] = vals.mean(axis=1)

    out = out.dropna(subset=["time", "value"]).sort_values("time").reset_index(drop=True)
    return out[["time", "value"]]

def to_daily_last(df: pd.DataFrame, display_tz: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date", "value"])
    d = df.copy()
    d["date"] = d["time"].dt.tz_convert(display_tz).dt.date
    last = d.groupby("date", as_index=False)["value"].last()
    return last[["date", "value"]]

def load_history(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["date", "value"])
    h = pd.read_csv(path)
    # 柔軟に吸収
    cols = [str(c).strip().lower() for c in h.columns]
    h.columns = cols
    # 'time' で持っている古い形式にも対応
    if "date" not in h.columns and "time" in h.columns:
        h["date"] = pd.to_datetime(h["time"], errors="coerce").dt.date
    elif "date" in h.columns:
        h["date"] = pd.to_datetime(h["date"], errors="coerce").dt.date
    else:
        h["date"] = pd.NaT
    if "value" not in h.columns:
        # 最後の数値列を value とみなす
        num_cols = [c for c in h.columns if c not in ("date", "time") and pd.api.types.is_numeric_dtype(h[c])]
        h["value"] = h[num_cols[-1]] if num_cols else pd.NA
    h = h.dropna(subset=["date", "value"])
    return h[["date", "value"]]

def save_history(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    log(f"written: {path} rows={len(df)}")

def main():
    key = get_key()
    raw_tz = "Asia/Tokyo"     # R-BANK9 は日本株
    display_tz = "Asia/Tokyo"

    intraday_csv = os.path.join(OUTPUT_DIR, f"{key}_intraday.csv")
    history_csv  = os.path.join(OUTPUT_DIR, f"{key}_history.csv")

    log(f"read intraday: {intraday_csv}")
    intraday = read_intraday(intraday_csv, raw_tz, display_tz)
    if intraday.empty:
        raise SystemExit("intraday empty – nothing to append")

    daily_last = to_daily_last(intraday, display_tz)

    hist = load_history(history_csv)
    merged = pd.concat([hist, daily_last], ignore_index=True)
    # 同一日を last で上書き
    merged = merged.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    save_history(merged, history_csv)

if __name__ == "__main__":
    main()
