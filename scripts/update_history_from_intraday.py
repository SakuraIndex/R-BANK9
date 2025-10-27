#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import os
import pandas as pd

INDEX_KEY = os.environ.get("INDEX_KEY", "rbank9")
OUT_DIR = Path(os.environ.get("OUT_DIR", "docs/outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

APPEND_TODAY_IF_MISSING = os.environ.get("APPEND_TODAY_IF_MISSING", "false").lower() == "true"
MARKET_TZ = os.environ.get("MARKET_TZ", "Asia/Tokyo")

intraday_csv = OUT_DIR / f"{INDEX_KEY}_intraday.csv"
history_csv  = OUT_DIR / f"{INDEX_KEY}_history.csv"

def read_two_col_csv(path: Path, ts_name="ts", val_name="val") -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=[ts_name, val_name])
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        return pd.DataFrame(columns=[ts_name, val_name])
    ts_col, val_col = df.columns[:2]
    df = df.rename(columns={ts_col: ts_name, val_col: val_name})
    # 時刻 → タイムゾーンつき（JST）
    ts = pd.to_datetime(df[ts_name], errors="coerce", utc=False)
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(MARKET_TZ)  # naive を JST とみなす
    else:
        ts = ts.dt.tz_convert(MARKET_TZ)
    df[ts_name] = ts
    df[val_name] = pd.to_numeric(df[val_name], errors="coerce")
    return df.dropna(subset=[ts_name, val_name]).sort_values(ts_name)

def main():
    intra = read_two_col_csv(intraday_csv)
    if intra.empty:
        print("[hist] intraday empty. skip")
        return

    # 日付ごと最後の値（≒終値）
    daily_last = intra.groupby(intra["ts"].dt.date, as_index=False).last(numeric_only=False)
    daily_last["ts"] = pd.to_datetime(daily_last["ts"]).dt.tz_localize(MARKET_TZ)  # keep tz

    hist = read_two_col_csv(history_csv)
    have_days = set(hist["ts"].dt.date) if not hist.empty else set()

    # まだ無い日だけ
    append = daily_last[~daily_last["ts"].dt.date.isin(have_days)].copy()

    # 「今日」をまだ持っていなくて、暫定で入れたい場合
    today = intra["ts"].dt.tz_convert(MARKET_TZ).dt.date.max()
    if APPEND_TODAY_IF_MISSING and today not in have_days:
        last_today = intra[intra["ts"].dt.date == today].tail(1)[["ts","val"]]
        if not last_today.empty:
            append = pd.concat([append, last_today], ignore_index=True)

    if append.empty:
        print("[hist] no new days to append")
        return

    new_hist = pd.concat([hist, append], ignore_index=True).sort_values("ts")
    # CSV は tz-naive（日時文字列）で保存
    out = new_hist.copy()
    out["ts"] = out["ts"].dt.tz_convert(MARKET_TZ).dt.tz_localize(None)
    out.to_csv(history_csv, index=False)
    print(f"[hist] appended {len(append)} day(s) -> {history_csv}")

if __name__ == "__main__":
    main()
