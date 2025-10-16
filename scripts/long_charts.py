#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R-BANK9 charts + stats
- 1d: 始値比の変化率(%)を描画（動的カラー）
- 7d/1m/1y: レベル値を描画
- pct_1d を stats.json に保存（scale="pct"）
"""

from pathlib import Path
import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =======================
# constants / paths
# =======================
INDEX_KEY = "rbank9"
MARKET_TZ = ZoneInfo("Asia/Tokyo")
OUTDIR    = Path("docs/outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

HISTORY_CSV  = OUTDIR / f"{INDEX_KEY}_history.csv"
INTRADAY_CSV = OUTDIR / f"{INDEX_KEY}_intraday.csv"

# =======================
# plotting style (dark)
# =======================
DARK_BG = "#0e0f13"
DARK_AX = "#0b0c10"
FG_TEXT = "#e7ecf1"
GRID    = "#2a2e3a"
GREEN   = "#28e07c"
RED     = "#ff4d4d"

def _apply(ax, title: str, ylabel: str) -> None:
    fig = ax.figure
    fig.set_size_inches(12, 7)
    fig.set_dpi(160)
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_AX)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.grid(color=GRID, alpha=0.6, linewidth=0.8)
    ax.tick_params(colors=FG_TEXT, labelsize=10)
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.set_title(title, color=FG_TEXT, fontsize=12)
    ax.set_xlabel("Time", color=FG_TEXT, fontsize=10)
    ax.set_ylabel(ylabel, color=FG_TEXT, fontsize=10)

def _save_line(x, y, out_png: Path, title: str, ylabel: str, color: str) -> None:
    fig, ax = plt.subplots()
    _apply(ax, title, ylabel)
    if len(x) > 0 and len(y) > 0:
        ax.plot(x, y, color=color, linewidth=1.6)
    fig.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

# =======================
# data helpers
# =======================
def _pick_index_column(df: pd.DataFrame) -> str:
    cand_names = {
        "rbank9", "r_bank9", "rbnk9", "rbank_9", "r_bank_9", "r-bank9",
        INDEX_KEY, INDEX_KEY.upper(), "R_BANK9", "RBANK9"
    }
    for c in df.columns:
        if c and c.strip().lower() in cand_names:
            return c
    return df.columns[-1]

def _load_df() -> pd.DataFrame:
    if INTRADAY_CSV.exists():
        df = pd.read_csv(INTRADAY_CSV, parse_dates=[0], index_col=0)
    elif HISTORY_CSV.exists():
        df = pd.read_csv(HISTORY_CSV, parse_dates=[0], index_col=0)
    else:
        raise FileNotFoundError("R-BANK9: CSV not found.")

    for c in list(df.columns):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(how="all")
    if df.index.tz is None:
        df.index = df.index.tz_localize(timezone.utc).tz_convert(MARKET_TZ)
    else:
        df.index = df.index.tz_convert(MARKET_TZ)
    return df

# =======================
# 1d chart (%)
# =======================
def _gen_1d_percent_png(df: pd.DataFrame, col: str) -> float | None:
    if df.empty:
        _save_line([], [], OUTDIR / f"{INDEX_KEY}_1d.png",
                   f"{INDEX_KEY.upper()} (1d %)", "Change (%)", FG_TEXT)
        return None

    local_idx = df.index
    last_day_midnight = local_idx[-1].normalize()
    day = df.loc[local_idx.normalize() == last_day_midnight, [col]].dropna()

    if len(day) < 2:
        _save_line([], [], OUTDIR / f"{INDEX_KEY}_1d.png",
                   f"{INDEX_KEY.upper()} (1d %)", "Change (%)", FG_TEXT)
        return None

    open_val = float(day[col].iloc[0])
    if not np.isfinite(open_val) or open_val == 0:
        return None

    series_pct = (day[col] / open_val - 1.0) * 100.0
    series_pct = series_pct.replace([np.inf, -np.inf], np.nan).dropna()

    if len(series_pct) == 0:
        _save_line([], [], OUTDIR / f"{INDEX_KEY}_1d.png",
                   f"{INDEX_KEY.upper()} (1d %)", "Change (%)", FG_TEXT)
        return None

    last_pct = float(series_pct.iloc[-1])
    color = GREEN if last_pct >= 0 else RED
    _save_line(series_pct.index, series_pct.values,
               OUTDIR / f"{INDEX_KEY}_1d.png",
               f"{INDEX_KEY.upper()} (1d %)", "Change (%)", color)
    return last_pct

# =======================
# 7d/1m/1y level charts
# =======================
def _gen_level_pngs(df: pd.DataFrame, col: str) -> None:
    for name, days in [("7d", 7), ("1m", 30), ("1y", 365)]:
        tail_df = df.last(f"{days}D")
        if len(tail_df) == 0:
            continue
        _save_line(tail_df.index, tail_df[col].values,
                   OUTDIR / f"{INDEX_KEY}_{name}.png",
                   f"{INDEX_KEY.upper()} ({name})", "Index / Value", FG_TEXT)

# =======================
# stats / markers
# =======================
def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def _write_stats(pct_1d: float | None) -> None:
    payload = {
        "index_key": INDEX_KEY,
        "pct_1d": None if pct_1d is None or not np.isfinite(pct_1d) else round(pct_1d, 6),
        "scale": "pct",
        "updated_at": _now_utc_iso(),
    }
    (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )

def _write_marker(pct_1d: float | None) -> None:
    marker = OUTDIR / f"{INDEX_KEY}_post_intraday.txt"
    if pct_1d is None or not np.isfinite(pct_1d):
        marker.write_text(f"{INDEX_KEY.upper()} 1d: N/A\n", encoding="utf-8")
    else:
        marker.write_text(f"{INDEX_KEY.upper()} 1d: {pct_1d:+.2f}%\n", encoding="utf-8")

# =======================
# main
# =======================
def main() -> None:
    df = _load_df()
    col = _pick_index_column(df)
    pct_1d = _gen_1d_percent_png(df, col)
    _gen_level_pngs(df, col)
    _write_stats(pct_1d)
    _write_marker(pct_1d)

if __name__ == "__main__":
    main()
