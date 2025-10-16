#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R-BANK9 (safe version): stable intraday % chart + stats
"""
from pathlib import Path
import json
from datetime import datetime, timezone
import pandas as pd
import matplotlib.pyplot as plt

INDEX_KEY = "rbank9"
OUTDIR = Path("docs/outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)
CSV = OUTDIR / f"{INDEX_KEY}_intraday.csv"

# ========== STYLE ==========
DARK_BG = "#0e0f13"
DARK_AX = "#0b0c10"
FG_TEXT = "#e7ecf1"
GRID = "#2a2e3a"
GREEN = "#28e07c"
RED = "#ff4d4d"

def _style(ax, title):
    fig = ax.figure
    fig.set_size_inches(12, 7)
    fig.set_dpi(160)
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_AX)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.grid(color=GRID, alpha=0.6)
    ax.tick_params(colors=FG_TEXT)
    ax.set_title(title, color=FG_TEXT, fontsize=12)
    ax.set_xlabel("Time", color=FG_TEXT)
    ax.set_ylabel("Change (%)", color=FG_TEXT)

# ========== MAIN LOGIC ==========
def load_rbank9_series() -> pd.Series:
    df = pd.read_csv(CSV, parse_dates=[0], index_col=0)
    df.columns = [c.strip() for c in df.columns]
    df = df.apply(pd.to_numeric, errors="coerce")

    # 優先列
    col = next((c for c in df.columns if c.upper() == "R_BANK9" or c.lower() == "rbank9"), None)
    s = df[col] if col else df.mean(axis=1, skipna=True)

    # 00:00行などゼロ・NaNのみの先頭行を除去
    s = s.dropna()
    s = s[s != 0]

    # 欠損を線形補完
    s = s.interpolate(limit_direction="both")

    return s

def compute_pct(s: pd.Series) -> pd.Series | None:
    if len(s) == 0:
        return None
    base = s.iloc[0]
    if base == 0 or pd.isna(base):
        return None
    pct = (s / base - 1.0) * 100.0
    pct = pct.dropna()
    pct = pct[~pct.isin([float("inf"), float("-inf")])]
    return pct

def plot_pct(pct: pd.Series, path: Path):
    color = GREEN if pct.iloc[-1] >= 0 else RED
    fig, ax = plt.subplots()
    _style(ax, f"{INDEX_KEY.upper()} (1d %)")
    ax.plot(pct.index, pct.values, color=color, linewidth=1.8)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

def write_stats(last_val: float | None):
    data = {
        "index_key": INDEX_KEY,
        "pct_1d": None if last_val is None else round(last_val, 6),
        "scale": "pct",
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00","Z")
    }
    (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    marker = OUTDIR / f"{INDEX_KEY}_post_intraday.txt"
    if last_val is None:
        marker.write_text(f"{INDEX_KEY.upper()} 1d: N/A\n", encoding="utf-8")
    else:
        marker.write_text(f"{INDEX_KEY.upper()} 1d: {last_val:+.2f}%\n", encoding="utf-8")

if __name__ == "__main__":
    s = load_rbank9_series()
    pct = compute_pct(s)
    if pct is None or len(pct) == 0:
        write_stats(None)
    else:
        plot_pct(pct, OUTDIR / f"{INDEX_KEY}_1d.png")
        write_stats(float(pct.iloc[-1]))
