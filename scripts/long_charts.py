#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
R-BANK9 long-term charts (7d / 1m / 1y)
自動列検出 + データ1行でも描画保証版
"""

import os
from pathlib import Path
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, AutoDateFormatter
from matplotlib.ticker import MaxNLocator

INDEX_KEY = os.environ.get("INDEX_KEY", "rbank9")
OUT_DIR = Path(os.environ.get("OUT_DIR", "docs/outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

def log(msg: str):
    print(f"[long] {msg}", flush=True)

# ───────────────────────────────
# CSV読み込み（列自動検出対応）
def load_history() -> pd.DataFrame:
    file = OUT_DIR / f"{INDEX_KEY}_history.csv"
    if not file.exists():
        log(f"{file} not found.")
        return pd.DataFrame()

    df = pd.read_csv(file)
    log(f"Loaded {len(df)} rows from {file.name}")

    if df.empty:
        return df

    # 日時列を推定
    time_candidates = [c for c in df.columns if any(k in c.lower() for k in ["ts", "time", "date"])]
    time_col = time_candidates[0] if time_candidates else df.columns[0]

    # 値列を推定
    val_candidates = [c for c in df.columns if any(k in c.lower() for k in ["val", "value", "price", "index"])]
    val_col = val_candidates[0] if val_candidates else df.columns[-1]

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")

    df = df.dropna(subset=[time_col, val_col])
    df = df.sort_values(time_col).drop_duplicates(subset=[time_col])

    if df.empty:
        log("No valid data after parsing.")
        return df

    # 最低限の形
    df = df.rename(columns={time_col: "ts", val_col: "val"})

    # 行が1つだけなら、描画のために擬似2点を追加
    if len(df) == 1:
        ts0 = df["ts"].iloc[0]
        val0 = df["val"].iloc[0]
        df = pd.DataFrame({
            "ts": [ts0 - timedelta(days=1), ts0],
            "val": [val0, val0]
        })
        log("Only 1 row -> duplicated to ensure plotting.")

    return df

# ───────────────────────────────
# 期間スライス
def slice_span(df: pd.DataFrame, days: int) -> pd.DataFrame:
    if df.empty:
        return df
    end = df["ts"].max()
    start = end - timedelta(days=days)
    sliced = df[df["ts"] >= start]
    if len(sliced) < 2:
        sliced = df.tail(2)
    return sliced

# ───────────────────────────────
# 描画共通
def apply_dark(ax, title: str):
    fig = ax.figure
    fig.set_size_inches(16, 8)
    fig.patch.set_facecolor("#111317")
    ax.set_facecolor("#111317")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors="#e8eef7")
    ax.yaxis.label.set_color("#e8eef7")
    ax.xaxis.label.set_color("#e8eef7")
    ax.title.set_color("#e8eef7")
    ax.grid(True, color="#2a2e3a", linestyle="-", alpha=0.5)
    ax.set_title(title, fontsize=22, fontweight="bold", pad=14)
    ax.set_xlabel("Time")
    ax.set_ylabel("Level (index)")

def plot_chart(df: pd.DataFrame, days: int):
    span = {7: "7d", 30: "1m", 365: "1y"}[days]
    out_file = OUT_DIR / f"{INDEX_KEY}_{span}.png"

    fig, ax = plt.subplots()
    apply_dark(ax, f"{INDEX_KEY.upper()} ({span} level)")

    ax.plot(df["ts"], df["val"], color="#00bfff", linewidth=2.5)
    ax.xaxis.set_major_locator(AutoDateLocator())
    ax.xaxis.set_major_formatter(AutoDateFormatter(AutoDateLocator()))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    vmin, vmax = df["val"].min(), df["val"].max()
    if vmin == vmax:
        pad = max(0.01, vmin * 0.02)
        ax.set_ylim(vmin - pad, vmax + pad)
    else:
        ax.set_ylim(vmin - (vmax - vmin) * 0.05, vmax + (vmax - vmin) * 0.05)

    fig.tight_layout()
    fig.savefig(out_file, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    log(f"Wrote {out_file.name}")

# ───────────────────────────────
def main():
    df = load_history()
    if df.empty:
        log("No data to plot.")
        return

    for d in [7, 30, 365]:
        sliced = slice_span(df, d)
        plot_chart(sliced, d)

    (OUT_DIR / "_last_run.txt").write_text(
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        encoding="utf-8",
    )

if __name__ == "__main__":
    main()
