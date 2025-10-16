#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R-BANK9 charts + stats
- 1d/7d/1m/1y を「％変化」で可視化
- 開始値→終了値で色を自動切替（上昇: GREEN / 下落: RED）
- stats.json (pct_1d) とマーカーは 1d グラフと同一ロジックで計算
"""
from pathlib import Path
import json
from datetime import datetime, timezone
import pandas as pd
import numpy as np
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

def _apply(ax, title: str, ylabel: str = "Change (%)") -> None:
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

# ------------------------
# data helpers
# ------------------------
def _pick_index_column(df: pd.DataFrame) -> str:
    """R-BANK9 列候補から推定。無ければ最後の列。"""
    cand = {
        "rbank9", "r_bank9", "rbnk9", "rbank_9", "r_bank_9", "r-bank9",
        INDEX_KEY, INDEX_KEY.upper(), "R_BANK9", "RBANK9"
    }
    for c in df.columns:
        if c.strip().lower() in cand:
            return c
    return df.columns[-1]

def _load_df() -> pd.DataFrame:
    """intraday 優先 → history。日時index・数値化・空行削除。"""
    if INTRADAY_CSV.exists():
        df = pd.read_csv(INTRADAY_CSV, parse_dates=[0], index_col=0)
    elif HISTORY_CSV.exists():
        df = pd.read_csv(HISTORY_CSV, parse_dates=[0], index_col=0)
    else:
        raise FileNotFoundError("R-BANK9: neither intraday nor history csv found.")
    for c in list(df.columns):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(how="all")
    return df.sort_index()

def _window_percent(df: pd.DataFrame, col: str, days: int) -> pd.DataFrame:
    """
    最後の時刻 end から days 日の窓を切り出し、先頭値基準の％変化に変換。
    """
    if df.empty:
        return pd.DataFrame(columns=["pct"], index=df.index)

    end = df.index.max()
    start = end - pd.Timedelta(days=days)
    win = df.loc[(df.index > start) & (df.index <= end), [col]].copy()
    win[col] = pd.to_numeric(win[col], errors="coerce")
    win = win.dropna()

    if len(win) == 0:
        # 窓にデータが無いときは N/A を返す
        return pd.DataFrame(columns=["pct"], index=win.index)

    first = win[col].iloc[0]
    if not np.isfinite(first) or first == 0:
        return pd.DataFrame(columns=["pct"], index=win.index)

    win["pct"] = (win[col] / first - 1.0) * 100.0
    return win[["pct"]]

def _choose_color(series: pd.Series) -> str:
    """系列の最初→最後で色を決定。"""
    if series is None or len(series) < 2:
        return FG_TEXT
    first = series.iloc[0]
    last  = series.iloc[-1]
    if not (np.isfinite(first) and np.isfinite(last)):
        return FG_TEXT
    return GREEN if last >= first else RED

def _save_pct(series: pd.Series, out_png: Path, title: str) -> None:
    """％系列を描画して保存。データ不足時は空キャンバスに軸のみ。"""
    fig, ax = plt.subplots()
    _apply(ax, title, ylabel="Change (%)")
    if series is not None and len(series) >= 1:
        color = _choose_color(series)
        ax.plot(series.index, series.values, color=color, linewidth=1.6)
    fig.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

# ------------------------
# charts
# ------------------------
def gen_pngs() -> None:
    df = _load_df()
    col = _pick_index_column(df)

    s1d = _window_percent(df, col, days=1)["pct"] if not df.empty else pd.Series(dtype=float)
    s7d = _window_percent(df, col, days=7)["pct"] if not df.empty else pd.Series(dtype=float)
    s1m = _window_percent(df, col, days=30)["pct"] if not df.empty else pd.Series(dtype=float)
    s1y = _window_percent(df, col, days=365)["pct"] if not df.empty else pd.Series(dtype=float)

    _save_pct(s1d, OUTDIR / f"{INDEX_KEY}_1d.png", f"{INDEX_KEY.upper()} (1d %)")
    _save_pct(s7d, OUTDIR / f"{INDEX_KEY}_7d.png", f"{INDEX_KEY.upper()} (7d %)")
    _save_pct(s1m, OUTDIR / f"{INDEX_KEY}_1m.png", f"{INDEX_KEY.upper()} (1m %)")
    _save_pct(s1y, OUTDIR / f"{INDEX_KEY}_1y.png", f"{INDEX_KEY.upper()} (1y %)")

# ------------------------
# stats (pct_1d) + marker
# ------------------------
def _now_utc_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )

def write_stats_and_marker() -> None:
    """
    1d グラフと同一ロジックで pct_1d を算出して出力。
    """
    df = _load_df()
    col = _pick_index_column(df)

    pct = None
    if not df.empty and col in df.columns:
        end = df.index.max()
        start = end - pd.Timedelta(days=1)
        day = df.loc[(df.index > start) & (df.index <= end), [col]].copy()
        day[col] = pd.to_numeric(day[col], errors="coerce")
        day = day.dropna()
        if len(day) >= 2:
            first = float(day[col].iloc[0])
            last  = float(day[col].iloc[-1])
            if first != 0 and np.isfinite(first) and np.isfinite(last):
                pct = (last / first - 1.0) * 100.0

    payload = {
        "index_key": INDEX_KEY,
        "pct_1d": None if pct is None else round(pct, 6),
        "scale": "pct",
        "updated_at": _now_utc_iso(),
    }
    (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )

    marker = OUTDIR / f"{INDEX_KEY}_post_intraday.txt"
    marker.write_text(
        f"{INDEX_KEY.upper()} 1d: {'N/A' if pct is None else f'{pct:+.2f}%'}\n",
        encoding="utf-8",
    )

# ------------------------
# main
# ------------------------
if __name__ == "__main__":
    gen_pngs()
    write_stats_and_marker()
