#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R-BANK9 charts + stats
- 先頭行(その日のCSV最初の時刻)を「基準時刻」に固定
- 先頭値が 0 / NaN / 非正 なら、先頭以降の最初の有効値で基準値を置換（時刻は先頭のまま）
- 1d/7d/1m/1y のPNGは 1dのみ %（Change from first-of-day）、他はレベル
- 線色は 1dのみ 基準→終値の増減で自動切替
- ダークテーマ
"""

from pathlib import Path
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------
# constants / paths
# ------------------------
INDEX_KEY = "rbank9"
OUTDIR = Path("docs/outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

HISTORY_CSV  = OUTDIR / f"{INDEX_KEY}_history.csv"
INTRADAY_CSV = OUTDIR / f"{INDEX_KEY}_intraday.csv"

# ------------------------
# plotting style (dark)
# ------------------------
DARK_BG = "#0e0f13"
DARK_AX = "#0b0c10"
FG_TEXT = "#e7ecf1"
GRID    = "#2a2e3a"
GREEN   = "#28e07c"   # 上昇
RED     = "#ff4d4d"   # 下落
NEUTRAL = "#cfd8dc"

def _apply(ax, title: str, y_label: str) -> None:
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
    ax.set_ylabel(y_label, color=FG_TEXT, fontsize=10)

def _save_line(df: pd.DataFrame, col: str, out_png: Path, title: str, color: str, y_label: str) -> None:
    fig, ax = plt.subplots()
    _apply(ax, title, y_label)
    ax.plot(df.index, df[col], color=color, linewidth=1.6)
    fig.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

# ------------------------
# data loading helpers
# ------------------------
def _pick_index_column(df: pd.DataFrame) -> str:
    """R-BANK9列名を推定（無ければ最後の列）。"""
    cand_names = {
        "rbank9", "r_bank9", "rbnk9", "rbank_9", "r_bank_9", "r-bank9",
        INDEX_KEY, INDEX_KEY.upper(), "R_BANK9", "RBANK9"
    }
    for c in df.columns:
        if c.strip().lower() in cand_names:
            return c
    return df.columns[-1]

def _load_df() -> pd.DataFrame:
    """intraday優先でロード。先頭列DatetimeIndex、数値化。"""
    if INTRADAY_CSV.exists():
        df = pd.read_csv(INTRADAY_CSV, parse_dates=[0], index_col=0)
    elif HISTORY_CSV.exists():
        df = pd.read_csv(HISTORY_CSV, parse_dates=[0], index_col=0)
    else:
        raise FileNotFoundError("R-BANK9: neither intraday nor history csv found.")
    df = df.sort_index()
    for c in list(df.columns):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ------------------------
# baseline (first row of the day) fixed
# ------------------------
def _first_row_of_day(df: pd.DataFrame) -> pd.DataFrame:
    """その日のデータだけ抽出（UTC基準の1日末尾まで）。"""
    # intraday想定なので、単純に最後の24h相当を採用
    # 直近のカレンダー日で切りたい場合は、必要に応じてtzを合わせて resample 等に変更
    return df.last("1D")

def _compute_1d_change_df(df_day: pd.DataFrame, col: str):
    """
    1d用: 先頭行の時刻を「基準時刻」に固定し、基準値で % 変換したシリーズを返す。
    先頭値が 0/NaN/非正なら先頭以降の最初の有効値を基準値に使用（時刻は最初のままと明記）。
    戻り値: (pct_df, basis_ts, basis_val, last_ts, last_val, used_fallback: bool)
    """
    if len(df_day) == 0 or col not in df_day.columns:
        return None, None, None, None, None, False

    basis_ts = df_day.index[0]
    basis_val = df_day[col].iloc[0]
    used_fallback = False

    # 無効な基準値なら、先頭以降の最初の有効値に置換（時刻は最初のまま）
    if pd.isna(basis_val) or basis_val <= 0:
        valid = df_day[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) == 0:
            return None, basis_ts, None, None, None, False
        basis_val = float(valid.iloc[0])
        used_fallback = True

    # % 変換
    pct = (df_day[col] / basis_val - 1.0) * 100.0
    pct_df = pd.DataFrame({col: pct})
    last_ts = df_day.index[-1]
    last_val = df_day[col].iloc[-1]
    return pct_df, basis_ts, float(basis_val), last_ts, float(last_val), used_fallback

# ------------------------
# chart generation
# ------------------------
def gen_pngs() -> tuple:
    df = _load_df()
    col = _pick_index_column(df)

    # 1d (%)
    df_day = _first_row_of_day(df[[col]].copy())
    pct_df, basis_ts, basis_val, last_ts, last_val, used_fallback = _compute_1d_change_df(df_day, col)

    if pct_df is not None:
        # 線色: 終値 vs 基準値
        color = NEUTRAL
        if basis_val is not None and last_val is not None:
            color = GREEN if last_val >= basis_val else RED

        out = OUTDIR / f"{INDEX_KEY}_1d.png"
        _save_line(
            pct_df, col, out,
            f"{INDEX_KEY.upper()} (1d %)",
            color,
            "Change (%)"
        )

    # 7d/1m/1y はレベル（元の値）でそのまま描画
    tail_7d = df.tail(7 * 1000)
    _save_line(tail_7d, col, OUTDIR / f"{INDEX_KEY}_7d.png", f"{INDEX_KEY.upper()} (7d)", NEUTRAL, "Index / Value")

    _save_line(df.tail(30 * 1000),  col, OUTDIR / f"{INDEX_KEY}_1m.png", f"{INDEX_KEY.upper()} (1m)", NEUTRAL, "Index / Value")
    _save_line(df.tail(365 * 1000), col, OUTDIR / f"{INDEX_KEY}_1y.png", f"{INDEX_KEY.upper()} (1y)", NEUTRAL, "Index / Value")

    return (basis_ts, basis_val, last_ts, last_val, pct_df[col].iloc[-1] if pct_df is not None else None, used_fallback)

# ------------------------
# stats (% from first-of-day) + marker
# ------------------------
def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def write_stats_and_marker() -> None:
    df = _load_df()
    col = _pick_index_column(df)
    df_day = _first_row_of_day(df[[col]].copy())
    pct_df, basis_ts, basis_val, last_ts, last_val, used_fallback = _compute_1d_change_df(df_day, col)

    pct_val = None
    if pct_df is not None:
        last_pct = float(pct_df[col].iloc[-1])
        pct_val = last_pct

    payload = {
        "index_key": INDEX_KEY,
        "pct_1d": None if pct_val is None else round(pct_val, 6),
        "scale": "pct",
        "updated_at": _now_utc_iso(),
    }
    (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    # human-readable marker
    marker = OUTDIR / f"{INDEX_KEY}_post_intraday.txt"
    if pct_val is None:
        marker.write_text(f"{INDEX_KEY.upper()} 1d: N/A\n", encoding="utf-8")
    else:
        note = ""
        if used_fallback:
            # 先頭値が無効で置換したことを明記
            # 例: basis 00:00 invalid→used first valid @ 00:05
            note = " (basis first-row invalid→used first valid)"
        marker.write_text(f"{INDEX_KEY.upper()} 1d: {pct_val:+.2f}%{note}\n", encoding="utf-8")

# ------------------------
# main
# ------------------------
if __name__ == "__main__":
    gen_pngs()
    write_stats_and_marker()
