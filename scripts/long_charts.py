#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R-BANK9 index chart generator
Generates 1d / 7d / 1m / 1y PNG charts + JSON stats marker.
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json

# ==========================================================
# ---- CONFIG ----
INDEX_KEY = "rbank9"
OUTDIR = Path("docs/outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)
HISTORY_CSV = OUTDIR / f"{INDEX_KEY}_history.csv"
INTRADAY_CSV = OUTDIR / f"{INDEX_KEY}_intraday.csv"
# ==========================================================

# ---- Dark theme style ----
DARK_BG = "#0e0f13"
DARK_AX = "#0b0c10"
FG_TEXT = "#e7ecf1"
LINE = "#2a2e3a"
RED = "#ff6b6b"


def _apply_common_style(ax, title):
    fig = ax.figure
    fig.set_size_inches(12, 7)
    fig.set_dpi(160)
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_AX)
    for sp in ax.spines.values():
        sp.set_color(LINE)
    ax.grid(color=LINE, alpha=0.35, linewidth=0.6)
    ax.tick_params(colors=FG_TEXT, labelsize=10)
    ax.yaxis.get_major_formatter().set_scientific(False)  # remove "1e13+..."
    ax.set_title(title, color=FG_TEXT, fontsize=12)
    ax.set_xlabel("Time", color=FG_TEXT, fontsize=10)
    ax.set_ylabel("Index Value", color=FG_TEXT, fontsize=10)


def save_line_png(df, col, out_png, title):
    """Draw a clean dark-mode line chart."""
    fig, ax = plt.subplots()
    _apply_common_style(ax, title)
    ax.plot(df.index, df[col], linewidth=1.6, color=RED)
    fig.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def load_latest_df():
    """Prefer intraday CSV if available, else fallback to history."""
    if INTRADAY_CSV.exists():
        df = pd.read_csv(INTRADAY_CSV, parse_dates=[0], index_col=0)
    elif HISTORY_CSV.exists():
        df = pd.read_csv(HISTORY_CSV, parse_dates=[0], index_col=0)
    else:
        raise FileNotFoundError("No data CSV found for R-BANK9")
    df = df.dropna()
    return df


def generate_all_charts():
    df = load_latest_df()
    col = df.columns[-1]
    save_line_png(df.tail(1000), col, OUTDIR / f"{INDEX_KEY}_1d.png", f"{INDEX_KEY.upper()} (1d)")
    save_line_png(df.tail(7 * 1000), col, OUTDIR / f"{INDEX_KEY}_7d.png", f"{INDEX_KEY.upper()} (7d)")
    save_line_png(df, col, OUTDIR / f"{INDEX_KEY}_1m.png", f"{INDEX_KEY.upper()} (1m)")
    save_line_png(df, col, OUTDIR / f"{INDEX_KEY}_1y.png", f"{INDEX_KEY.upper()} (1y)")
    print("[INFO] PNG charts generated.")


def write_stats_and_marker():
    df = load_latest_df()
    col = df.columns[-1]
    pct = None
    if len(df) >= 2:
        pct = (float(df[col].iloc[-1]) / float(df[col].iloc[0]) - 1.0) * 100.0
    now_iso = pd.Timestamp.utcnow().isoformat() + "Z"

    stats = {
        "index_key": INDEX_KEY,
        "pct_1d": pct,
        "scale": "pct",
        "updated_at": now_iso,
    }
    (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(json.dumps(stats, ensure_ascii=False), encoding="utf-8")

    if pct is not None:
        marker = f"{INDEX_KEY.upper()} 1d: {pct:+.2f}%"
        (OUTDIR / f"{INDEX_KEY}_post_intraday.txt").write_text(marker + "\n", encoding="utf-8")
        print("[INFO] Marker written:", marker)
    else:
        print("[WARN] Could not compute 1d pct")


if __name__ == "__main__":
    try:
        generate_all_charts()
        write_stats_and_marker()
    except Exception as e:
        print("[ERROR] Failed:", e)
