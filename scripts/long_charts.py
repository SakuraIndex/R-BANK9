#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R-BANK9 charts + stats
- データ入力  : intraday/history は「指数レベル（累積値）」を想定
- 可視化      : 各ウィンドウの先頭値で正規化した「変化率(%)」で描画
- 騰落率      : 当日（1D）先頭→最後の%変化を stats.json に書き出し
- 線色        : 先頭→最後で上昇=緑 / 下落=赤 を自動切り替え
- ダークテーマ
"""
from pathlib import Path
import json
from datetime import datetime, timezone
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
GREEN   = "#28e07c"  # 上昇
RED     = "#ff4d4d"  # 下落
MUTED   = "#9aa3ad"

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

def _color_by_first_last(series: pd.Series) -> str:
    if len(series) < 2:
        return MUTED
    first, last = float(series.iloc[0]), float(series.iloc[-1])
    return GREEN if last >= first else RED

def _save_pct(df_pct: pd.DataFrame, col: str, out_png: Path, title: str) -> None:
    """
    すでに % 変換済みの df_pct[col] を描画して保存。
    線色は先頭→最後の増減で自動決定。
    """
    fig, ax = plt.subplots()
    _apply(ax, title, ylabel="Change (%)")
    color = _color_by_first_last(df_pct[col])
    ax.plot(df_pct.index, df_pct[col], color=color, linewidth=1.6)
    fig.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

# ------------------------
# data loading helpers
# ------------------------
def _pick_index_column(df: pd.DataFrame) -> str:
    """
    優先順位で R-BANK9 の列を決定。無ければ最後の列を使う。
    """
    cand_names = {
        "rbank9", "r_bank9", "rbnk9", "rbank_9", "r_bank_9", "r-bank9",
        INDEX_KEY, INDEX_KEY.upper(), "R_BANK9", "RBANK9"
    }
    for c in df.columns:
        if c and c.strip().lower() in cand_names:
            return c
    return df.columns[-1]

def _load_df() -> pd.DataFrame:
    """
    intraday があれば intraday 優先、無ければ history。
    先頭列を DatetimeIndex にして数値化、全NA行は除去。
    """
    if INTRADAY_CSV.exists():
        df = pd.read_csv(INTRADAY_CSV, parse_dates=[0], index_col=0)
    elif HISTORY_CSV.exists():
        df = pd.read_csv(HISTORY_CSV, parse_dates=[0], index_col=0)
    else:
        raise FileNotFoundError("R-BANK9: neither intraday nor history csv found.")
    for c in list(df.columns):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(how="all")
    return df

def _window_to_pct(df: pd.DataFrame, col: str, period: str) -> pd.DataFrame:
    """
    指定 period（例: '1D','7D','30D','365D'）の範囲に絞り、先頭値基準の % 変化に変換。
      pct = (value / first - 1) * 100
    データ点が少なければ素直に末尾 1000 行を対象にする。
    """
    sub = df.last(period).copy()
    if len(sub) < 2:
        sub = df.tail(1000).copy()
    first = sub[col].iloc[0]
    if pd.isna(first) or first == 0:
        # 極端ケースはゼロ除算回避のため、そのまま返す（実線はMUTED色になる）
        return sub
    sub[col] = (sub[col] / first - 1.0) * 100.0
    return sub

# ------------------------
# chart generation
# ------------------------
def gen_pngs() -> None:
    df = _load_df()
    col = _pick_index_column(df)

    # 各ウィンドウで % に変換して保存
    d1  = _window_to_pct(df, col, "1D")
    d7  = _window_to_pct(df, col, "7D")
    d30 = _window_to_pct(df, col, "30D")
    d365= _window_to_pct(df, col, "365D")

    _save_pct(d1,   col, OUTDIR / f"{INDEX_KEY}_1d.png", f"{INDEX_KEY.upper()} (1d %)")
    _save_pct(d7,   col, OUTDIR / f"{INDEX_KEY}_7d.png", f"{INDEX_KEY.upper()} (7d %)")
    _save_pct(d30,  col, OUTDIR / f"{INDEX_KEY}_1m.png", f"{INDEX_KEY.upper()} (1m %)")
    _save_pct(d365, col, OUTDIR / f"{INDEX_KEY}_1y.png", f"{INDEX_KEY.upper()} (1y %)")

# ------------------------
# stats (1d %) + marker writers
# ------------------------
def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def write_stats_and_marker() -> None:
    """
    当日の % 騰落率を算出し、stats.json と post_intraday.txt に出力。
    """
    df = _load_df()
    col = _pick_index_column(df)

    pct = None
    day = df.last("1D")
    if len(day) < 2:
        day = df.tail(1000)
    if len(day) >= 2:
        first = float(day[col].iloc[0])
        last  = float(day[col].iloc[-1])
        if first != 0 and pd.notna(first) and pd.notna(last):
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
    if pct is None:
        marker.write_text(f"{INDEX_KEY.upper()} 1d: N/A\n", encoding="utf-8")
    else:
        marker.write_text(f"{INDEX_KEY.upper()} 1d: {pct:+.2f}%\n", encoding="utf-8")

# ------------------------
# main
# ------------------------
if __name__ == "__main__":
    gen_pngs()
    write_stats_and_marker()
