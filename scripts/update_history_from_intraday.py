#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
intraday → history 追記ユーティリティ
- docs/outputs/rbank9_intraday.csv から日付ごとの「終値」(同日の最後の行)を取り、
  docs/outputs/rbank9_history.csv に未登録日だけ追記する。
- 既存のカラム名が何であっても、先頭列=時刻、2列目=値 として扱う。
"""

from pathlib import Path
import pandas as pd

INDEX_KEY = "rbank9"
OUT_DIR = Path("docs/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

intraday_csv = OUT_DIR / f"{INDEX_KEY}_intraday.csv"
history_csv  = OUT_DIR / f"{INDEX_KEY}_history.csv"

def load_intraday():
    if not intraday_csv.exists():
        print(f"[hist] no intraday: {intraday_csv}")
        return pd.DataFrame()
    df = pd.read_csv(intraday_csv)
    if df.shape[1] < 2:
        print(f"[hist] invalid intraday shape: {intraday_csv}")
        return pd.DataFrame()
    ts_col, val_col = df.columns[:2]
    df = df.rename(columns={ts_col:"ts", val_col:"val"})
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=False)
    df["val"] = pd.to_numeric(df["val"], errors="coerce")
    df = df.dropna().sort_values("ts")
    return df

def load_history():
    if not history_csv.exists():
        return pd.DataFrame(columns=["ts","val"])
    df = pd.read_csv(history_csv)
    if df.shape[1] < 2:
        return pd.DataFrame(columns=["ts","val"])
    ts_col, val_col = df.columns[:2]
    df = df.rename(columns={ts_col:"ts", val_col:"val"})
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=False)
    df["val"] = pd.to_numeric(df["val"], errors="coerce")
    df = df.dropna().sort_values("ts")
    return df

def main():
    intra = load_intraday()
    if intra.empty:
        print("[hist] intraday empty, skip")
        return

    # 日単位で最後の値（≒終値）を採用
    daily_last = intra.groupby(intra["ts"].dt.date, as_index=False).last()
    daily_last["ts"] = pd.to_datetime(daily_last["ts"])

    hist = load_history()

    # 既存 history にない日付だけ追加
    have = set(hist["ts"].dt.date) if not hist.empty else set()
    append_rows = daily_last[~daily_last["ts"].dt.date.isin(have)].copy()

    if append_rows.empty:
        print("[hist] no new days to append")
        return

    new_hist = pd.concat([hist, append_rows], ignore_index=True).sort_values("ts")
    new_hist.to_csv(history_csv, index=False)
    print(f"[hist] appended {len(append_rows)} day(s) -> {history_csv}")

if __name__ == "__main__":
    main()
