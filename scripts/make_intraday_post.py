#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Intraday CSV を読み、当日 09:00–15:30(JST) のセッションに基づく 1d 騰落率と
テキスト/JSON 出力を作成するユーティリティ。

使い方（例）:
python scripts/make_intraday_post.py \
  --index-key rbank9 \
  --csv docs/outputs/rbank9_intraday.csv \
  --out-json docs/outputs/rbank9_stats.json \
  --out-text docs/outputs/rbank9_post_intraday.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, time, timezone, timedelta

import pandas as pd

JST = timezone(timedelta(hours=9))
SESSION_LABEL = "09:00–15:30 JST"
SESSION_START = time(9, 0)
SESSION_END = time(15, 30)
EPS = 1e-6  # 分母ゼロ/極小値ガード


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--index-key", required=True)
    p.add_argument("--csv", required=True, help="intraday CSV (ts, value の先頭2列を想定)")
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-text", required=True)
    return p.parse_args()


def load_intraday(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError("CSV は少なくとも2列（時刻/値）を含む必要があります")
    ts_col, val_col = df.columns[:2]
    df = df.rename(columns={ts_col: "ts", val_col: "val"}).copy()

    # できるだけ JST として解釈（tz 無しなら JST を付与）
    ts = pd.to_datetime(df["ts"], errors="coerce", utc=False)
    # pandas Timestamp に tz がない場合は JST を付与
    ts = ts.dt.tz_localize(JST, nonexistent="shift_forward", ambiguous="NaT", errors="coerce") \
         .fillna(pd.to_datetime(df["ts"], errors="coerce", utc=True))  # 念のためのフォールバック
    df["ts"] = ts
    df = df.dropna(subset=["ts", "val"]).sort_values("ts").reset_index(drop=True)
    return df


def session_filter(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    d = df["ts"].dt.tz_convert(JST)
    mask = (d.dt.time >= SESSION_START) & (d.dt.time <= SESSION_END)
    sdf = df.loc[mask].copy()
    # 当日で絞る（最後の行の日付を採用）
    if not sdf.empty:
        last_day = sdf["ts"].dt.tz_convert(JST).dt.normalize().iloc[-1]
        sdf = sdf[sdf["ts"].dt.tz_convert(JST).dt.normalize() == last_day].copy()
    return sdf


def pick_open_at_9(df: pd.DataFrame) -> float | None:
    if df.empty:
        return None
    # 09:00 以降の最初の値（なければ全体の最初）
    jst = df["ts"].dt.tz_convert(JST)
    after_open = df[jst.dt.time >= SESSION_START]
    if not after_open.empty:
        return float(after_open.iloc[0]["val"])
    return float(df.iloc[0]["val"])


def calc_pct(base: float, close: float) -> float:
    denom = max(abs(base), EPS)
    return (close - base) / denom * 100.0


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    out_json = Path(args.out_json)
    out_text = Path(args.out_text)

    df = load_intraday(csv_path)
    df_sess = session_filter(df)

    if df_sess.empty:
        # セッションが該当しない場合は null / N/A を出力
        payload = {
            "index_key": args.index_key,
            "pct_1d": None,
            "delta_level": None,
            "scale": "percent",
            "basis": "no_session",
            "session": SESSION_LABEL,
            "updated_at": datetime.now(tz=JST).isoformat(timespec="seconds"),
        }
        out_json.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        out_text.write_text(
            f"{args.index_key.upper()} 1d: Δ=N/A (level) A%=N/A "
            f"(basis=no_session sess={SESSION_LABEL})\n",
            encoding="utf-8",
        )
        return

    base = pick_open_at_9(df_sess)
    close = float(df_sess.iloc[-1]["val"])
    pct = calc_pct(base, close)
    delta = close - base

    start_ts = df_sess.iloc[0]["ts"].astimezone(JST)
    end_ts = df_sess.iloc[-1]["ts"].astimezone(JST)

    payload = {
        "index_key": args.index_key,
        "pct_1d": pct,
        "delta_level": delta,
        "scale": "percent",
        "basis": "open@09:00",
        "session": SESSION_LABEL,
        "updated_at": datetime.now(tz=JST).isoformat(timespec="seconds"),
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    line = (
        f"{args.index_key.upper()} 1d: Δ={delta:+.6f} (level) "
        f"A%={pct:+.2f}% (basis=open@09:00 sess={SESSION_LABEL} "
        f"valid={start_ts.strftime('%Y-%m-%d %H:%M:%S')}->{end_ts.strftime('%Y-%m-%d %H:%M:%S')})\n"
    )
    out_text.write_text(line, encoding="utf-8")


if __name__ == "__main__":
    main()
