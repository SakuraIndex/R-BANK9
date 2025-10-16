#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R-BANK9 charts + stats (daily % scale, dark theme, color-coded)
"""
from pathlib import Path
import json
from datetime import datetime, timezone
import pandas as pd
import matplotlib.pyplot as plt

INDEX_KEY = "rbank9"
OUTDIR = Path("docs/outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

HISTORY_CSV  = OUTDIR / f"{INDEX_KEY}_history.csv"
INTRADAY_CSV = OUTDIR / f"{INDEX_KEY}_intraday.csv"

DARK_BG = "#0e0f13"
DARK_AX = "#0b0c10"
FG_TEXT = "#e7ecf1"
GRID    = "#2a2e3a"
GREEN   = "#28e07c"
RED     = "#ff4d4d"

def _apply(ax, title: str):
    fig = ax.figure
    fig.set_size_inches(12, 7)
    fig.set_dpi(160)
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_AX)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.grid(color=GRID, alpha=0.6, linewidth=0.8)
    ax.tick_params(colors=FG_TEXT, labelsize=10)
    ax.set_title(title, color=FG_TEXT, fontsize=12)
    ax.set_xlabel("Time", color=FG_TEXT, fontsize=10)
    ax.set_ylabel("Change (%)", color=FG_TEXT, fontsize=10)

def _save(df, col, out_png, title):
    if len(df) < 2:
        color = FG_TEXT
    else:
        color = GREEN if df[col].iloc[-1] >= 0 else RED
    fig, ax = plt.subplots()
    _apply(ax, title)
    ax.plot(df.index, df[col], color=color, linewidth=1.6)
    fig.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

def _pick_index_column(df):
    cand_names = {"rbank9", "r_bank9", "rbnk9", "rbank_9", "r-bank9"}
    for c in df.columns:
        if c.strip().lower() in cand_names:
            return c
    return df.columns[-1]

def _load_df():
    if INTRADAY_CSV.exists():
        df = pd.read_csv(INTRADAY_CSV, parse_dates=[0], index_col=0)
    elif HISTORY_CSV.exists():
        df = pd.read_csv(HISTORY_CSV, parse_dates=[0], index_col=0)
    else:
        raise FileNotFoundError("No data CSV found.")
    for c in list(df.columns):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(how="all")
    return df

def gen_pngs():
    df = _load_df()
    col = _pick_index_column(df)

    # === normalize to daily % scale ===
    first_val = df[col].iloc[0]
    df[col] = (df[col] / first_val - 1.0) * 100.0

    tail_1d = df.tail(1000)
    tail_7d = df.tail(7 * 1000)

    _save(tail_1d, col, OUTDIR / f"{INDEX_KEY}_1d.png", f"{INDEX_KEY.upper()} (1d %)")
    _save(tail_7d, col, OUTDIR / f"{INDEX_KEY}_7d.png", f"{INDEX_KEY.upper()} (7d %)")
    _save(df.tail(30 * 1000), col, OUTDIR / f"{INDEX_KEY}_1m.png", f"{INDEX_KEY.upper()} (1m %)")
    _save(df.tail(365 * 1000), col, OUTDIR / f"{INDEX_KEY}_1y.png", f"{INDEX_KEY.upper()} (1y %)")

def _now_utc_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def write_stats_and_marker():
    df = _load_df()
    col = _pick_index_column(df)
    first = df[col].iloc[0]
    last = df[col].iloc[-1]
    pct = (last / first - 1.0) * 100.0

    payload = {
        "index_key": INDEX_KEY,
        "pct_1d": round(pct, 6),
        "scale": "pct",
        "updated_at": _now_utc_iso(),
    }
    (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    marker = OUTDIR / f"{INDEX_KEY}_post_intraday.txt"
    marker.write_text(f"{INDEX_KEY.upper()} 1d: {pct:+.2f}%\n", encoding="utf-8")

if __name__ == "__main__":
    gen_pngs()
    write_stats_and_marker()
