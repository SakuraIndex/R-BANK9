# scripts/update_history.py
# -*- coding: utf-8 -*-
"""
Append today's close-like value into docs/outputs/<index>_history.csv as `date,value`.

- Reads docs/outputs/<index>_intraday.csv
- Picks the last numeric value as "close" (sorted by time if time exists)
- Appends one row per day (JST) if not already appended
"""

import os
from datetime import datetime
import pandas as pd
import pytz

JP_TZ = pytz.timezone("Asia/Tokyo")
OUTPUTS_DIR = os.path.join("docs", "outputs")

def _lower(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def main():
    index_key = os.environ.get("INDEX_KEY", "rbank9").strip().lower()
    intraday_csv = os.path.join(OUTPUTS_DIR, f"{index_key}_intraday.csv")
    history_csv  = os.path.join(OUTPUTS_DIR, f"{index_key}_history.csv")

    if not os.path.exists(intraday_csv):
        print(f"[WARN] no intraday file: {intraday_csv}")
        return

    df = pd.read_csv(intraday_csv)
    df = _lower(df)

    if "value" not in df.columns:
        print("[WARN] intraday.csv has no 'value' column")
        return

    # 値列を数値化して欠損を落とす
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    if df.empty:
        print("[WARN] all 'value' were NaN")
        return

    # time があればそれでソートして最後を終値扱い
    if "time" in df.columns:
        # どちらが来ても事故らないように to_datetime(utc=True) → JST 変換
        t = pd.to_datetime(df["time"], errors="coerce", utc=True)
        # naive が来たら JST ローカライズ
        na_mask = t.dt.tz is None
        if na_mask:
            # pandas の tz 判定は seriesごと、ここは安全にローカライズ
            t = pd.to_datetime(df["time"], errors="coerce").dt.tz_localize(JP_TZ).dt.tz_convert("UTC")
        df["_t"] = t.dt.tz_convert(JP_TZ)
        df = df.dropna(subset=["_t"]).sort_values("_t")
    # 最終行の value を採用
    end_value = float(df["value"].iloc[-1])

    today = datetime.now(JP_TZ).date()

    if os.path.exists(history_csv):
        hist = pd.read_csv(history_csv)
        hist = _lower(hist)
        if "date" not in hist.columns or "value" not in hist.columns:
            hist = pd.DataFrame(columns=["date", "value"])
    else:
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        hist = pd.DataFrame(columns=["date", "value"])

    # 既に今日が入っていればスキップ（上書きしたい場合は置換に変更OK）
    if not hist.empty and str(today) in hist["date"].astype(str).values:
        print(f"[INFO] already exists: {today}")
        return

    hist.loc[len(hist)] = [today, end_value]
    hist = hist.sort_values("date")
    hist.to_csv(history_csv, index=False)
    print(f"[OK] appended {today} -> {end_value}")

if __name__ == "__main__":
    main()
