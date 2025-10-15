#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R-BANK9 charts + stats
"""
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

INDEX_KEY = "rbank9"
OUTDIR = Path("docs/outputs"); OUTDIR.mkdir(parents=True, exist_ok=True)
HISTORY_CSV = OUTDIR / f"{INDEX_KEY}_history.csv"
INTRADAY_CSV = OUTDIR / f"{INDEX_KEY}_intraday.csv"

DARK_BG = "#0e0f13"; DARK_AX = "#0b0c10"; FG_TEXT = "#e7ecf1"
GRID = "#2a2e3a"; RED = "#ff6b6b"

def _apply(ax, title):
    fig = ax.figure; fig.set_size_inches(12, 7); fig.set_dpi(160)
    fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(DARK_AX)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.grid(color=GRID, alpha=0.6, linewidth=0.8)
    ax.tick_params(colors=FG_TEXT, labelsize=10)
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.set_title(title, color=FG_TEXT, fontsize=12)
    ax.set_xlabel("Time", color=FG_TEXT, fontsize=10)
    ax.set_ylabel("Index / Value", color=FG_TEXT, fontsize=10)

def _save(df, col, out_png, title):
    fig, ax = plt.subplots()
    _apply(ax, title)
    ax.plot(df.index, df[col], color=RED, linewidth=1.6)
    fig.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

def _load_df():
    if INTRADAY_CSV.exists():
        df = pd.read_csv(INTRADAY_CSV, parse_dates=[0], index_col=0)
    elif HISTORY_CSV.exists():
        df = pd.read_csv(HISTORY_CSV, parse_dates=[0], index_col=0)
    else:
        raise FileNotFoundError("no csv")
    return df.dropna()

def gen_pngs():
    df = _load_df(); col = df.columns[-1]
    _save(df.tail(1000), col, OUTDIR / f"{INDEX_KEY}_1d.png", f"{INDEX_KEY.upper()} (1d)")
    _save(df.tail(7*1000), col, OUTDIR / f"{INDEX_KEY}_7d.png", f"{INDEX_KEY.upper()} (7d)")
    _save(df, col, OUTDIR / f"{INDEX_KEY}_1m.png", f"{INDEX_KEY.upper()} (1m)")
    _save(df, col, OUTDIR / f"{INDEX_KEY}_1y.png", f"{INDEX_KEY.upper()} (1y)")

def write_stats():
    """
    R-BANK9 は intraday が %単位。
    → 騰落率[%] = last_value
    """
    df = _load_df(); col = df.columns[-1]
    pct = float(df[col].iloc[-1]) if len(df) else None

    payload = {
        "index_key": INDEX_KEY,
        "pct_1d": None if pct is None else pct,
        "scale": "pct",
        "updated_at": pd.Timestamp.utcnow().isoformat(timespec="seconds") + "Z",
    }
    (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    if pct is not None:
        (OUTDIR / f"{INDEX_KEY}_post_intraday.txt").write_text(
            f"{INDEX_KEY.upper()} 1d: {pct:+.2f}%\n", encoding="utf-8"
        )

if __name__ == "__main__":
    gen_pngs()
    write_stats()
