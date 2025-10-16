#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R-BANK9 long-term charts & stats
 - 1dは「％表示（基準＝CSV先頭の時刻を固定）」に統一
 - 先頭値が 0 / NaN / 非正のときは、先頭以降の最初の有効値でフォールバック
 - チャート色は％の終値で動的切替（>=0: GREEN, <0: RED）
 - stats.json / post_intraday.txt はチャートと同一ロジックで算出し整合
"""

from pathlib import Path
import os
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

# env（ワークフローから渡される想定。なければJSTデフォルト）
MARKET_TZ = os.environ.get("MARKET_TZ", "Asia/Tokyo")

# ------------------------
# plotting style (dark)
# ------------------------
DARK_BG = "#0e0f13"
DARK_AX = "#0b0c10"
FG_TEXT = "#e7ecf1"
GRID    = "#2a2e3a"
GREEN   = "#28e07c"   # 上昇
RED     = "#ff4d4d"   # 下落

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

def _plot_save(x, y, out_png: Path, title: str) -> None:
    # 1d％の色判定：終値 >= 0 ならGREEN、<0 ならRED
    color = FG_TEXT
    if len(y) >= 1:
        last = float(y[-1])
        color = GREEN if last >= 0 else RED

    fig, ax = plt.subplots()
    _apply(ax, title, "Change (%)")
    ax.plot(x, y, color=color, linewidth=1.8)
    fig.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

# ------------------------
# data loading helpers
# ------------------------
def _load_df() -> pd.DataFrame:
    """
    intraday があれば intraday 優先、無ければ history。
    先頭列を DatetimeIndex にし、数値に変換、全NaN行を除去。
    """
    if INTRADAY_CSV.exists():
        df = pd.read_csv(INTRADAY_CSV, parse_dates=[0], index_col=0)
    elif HISTORY_CSV.exists():
        df = pd.read_csv(HISTORY_CSV, parse_dates=[0], index_col=0)
    else:
        raise FileNotFoundError("R-BANK9: neither intraday nor history csv found.")

    # 数値化とクレンジング
    for c in list(df.columns):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    return df

def _pick_index_column(df: pd.DataFrame) -> str:
    """
    優先順位で R-BANK9 の列を決定。無ければ最後の列を使う。
    """
    cand_names = {
        "rbank9", "r_bank9", "rbnk9", "rbank_9", "r_bank_9", "r-bank9",
        INDEX_KEY, INDEX_KEY.upper(), "R_BANK9", "RBANK9"
    }
    low = {c.strip().lower(): c for c in df.columns}
    for key in list(low.keys()):
        if key in cand_names:
            return low[key]
    return df.columns[-1]

def _ensure_market_tz(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    index を MARKET_TZ に揃える。
    すでにtz-awareなら tz_convert、naive なら UTC 仮定で tz_localize → tz_convert。
    """
    if index.tz is None:
        idx = index.tz_localize("UTC").tz_convert(MARKET_TZ)
    else:
        idx = index.tz_convert(MARKET_TZ)
    return idx

def _slice_latest_local_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    ローカル市場TZに変換し、最新日の行のみ抽出。
    """
    df = df.copy()
    df.index = _ensure_market_tz(df.index)
    if len(df.index) == 0:
        return df
    latest_date = df.index[-1].date()
    return df[df.index.date == latest_date]

# ------------------------
# percent transform with robust basis
# ------------------------
def _compute_pct_fixed_basis(df_win: pd.DataFrame, col: str):
    """
    %変換：基準時刻はウィンドウの先頭（時刻固定）。
    先頭値が 0/NaN/非正→先頭以降で最初の有効な正値を基準値に採用（used_fallback=True）。
    戻り値: (pct_series, basis_ts, basis_val, used_fallback)
    """
    if len(df_win) == 0 or col not in df_win.columns:
        return None, None, None, False

    basis_ts = df_win.index[0]
    raw_val = df_win[col].iloc[0]
    basis_val = raw_val
    used_fallback = False

    # 無効な基準値ならフォールバック
    if pd.isna(raw_val) or raw_val <= 0:
        valid = df_win[col].replace([np.inf, -np.inf], np.nan).dropna()
        valid = valid[valid > 0]
        if len(valid) == 0:
            return None, basis_ts, None, False
        basis_val = float(valid.iloc[0])
        used_fallback = True

    arr = pd.to_numeric(df_win[col], errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = (arr / basis_val - 1.0) * 100.0
    pct = pct.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return pct, basis_ts, float(basis_val), used_fallback

# ------------------------
# chart generation
# ------------------------
def gen_pngs() -> None:
    df = _load_df()
    col = _pick_index_column(df)

    # 1d（最新日）％
    day = _slice_latest_local_day(df[[col]])
    pct_1d, _, _, _ = _compute_pct_fixed_basis(day, col)
    if pct_1d is not None and len(pct_1d) > 0:
        _plot_save(pct_1d.index, pct_1d.values,
                   OUTDIR / f"{INDEX_KEY}_1d.png", f"{INDEX_KEY.upper()} (1d %)")

    # 7d / 1m / 1y も％に統一（先頭固定＋フォールバック）
    # 期間は単純tailでOK（インデックスの頻度不問）
    def _save_window_pct(tail_n: int, label: str, out_name: str):
        win = df[[col]].tail(tail_n)
        win.index = _ensure_market_tz(win.index)
        pct, _, _, _ = _compute_pct_fixed_basis(win, col)
        if pct is not None and len(pct) > 0:
            _plot_save(pct.index, pct.values, OUTDIR / out_name, f"{INDEX_KEY.upper()} ({label} %)")

    _save_window_pct(7 * 24 * 60,  "7d", f"{INDEX_KEY}_7d.png")
    _save_window_pct(30 * 24 * 60, "1m", f"{INDEX_KEY}_1m.png")
    _save_window_pct(365 * 24 * 60, "1y", f"{INDEX_KEY}_1y.png")

# ------------------------
# stats (1d pct) + marker writers
# ------------------------
def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def write_stats_and_marker() -> None:
    """
    1d（最新日のウィンドウ）を％に変換した終値で統計を作る。
    チャートとまったく同じロジック（先頭固定＋フォールバック）を使用。
    """
    df = _load_df()
    col = _pick_index_column(df)
    day = _slice_latest_local_day(df[[col]])
    pct_series, basis_ts, basis_val, used_fallback = _compute_pct_fixed_basis(day, col)

    pct_val = None
    if pct_series is not None and len(pct_series) > 0:
        v = float(pct_series.iloc[-1])
        pct_val = 0.0 if not np.isfinite(v) else v

    # JSON（サイト向け）
    payload = {
        "index_key": INDEX_KEY,
        "pct_1d": None if pct_val is None else round(pct_val, 6),
        "scale": "pct",
        "updated_at": _now_utc_iso(),
    }
    (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )

    # 人間可読マーカー
    marker = OUTDIR / f"{INDEX_KEY}_post_intraday.txt"
    if pct_val is None:
        marker.write_text(f"{INDEX_KEY.upper()} 1d: N/A\n", encoding="utf-8")
    else:
        note = ""
        if used_fallback:
            note = " (basis first-row invalid→used first valid)"
        marker.write_text(f"{INDEX_KEY.upper()} 1d: {pct_val:+.2f}%{note}\n", encoding="utf-8")

# ------------------------
# main
# ------------------------
if __name__ == "__main__":
    gen_pngs()
    write_stats_and_marker()
