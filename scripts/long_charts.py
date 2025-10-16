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
import matplotlib.pyplot as plt

# =======================
# constants / paths
# =======================
INDEX_KEY = "rbank9"
MARKET_TZ = ZoneInfo("Asia/Tokyo")          # R-BANK9 は国内銘柄ベース
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
GREEN   = "#28e07c"     # 上昇
RED     = "#ff4d4d"     # 下落

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
    ax.plot(x, y, color=color, linewidth=1.6)
    fig.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

# =======================
# data loading helpers
# =======================
def _pick_index_column(df: pd.DataFrame) -> str:
    """
    R-BANK9 列を推定。既知の候補が無ければ最後の列。
    """
    cand_names = {
        "rbank9", "r_bank9", "rbnk9", "rbank_9", "r_bank_9", "r-bank9",
        INDEX_KEY, INDEX_KEY.upper(), "R_BANK9", "RBANK9"
    }
    for c in df.columns:
        if c is None:
            continue
        if c.strip().lower() in cand_names:
            return c
    return df.columns[-1]

def _load_df() -> pd.DataFrame:
    """
    intraday があれば intraday 優先、無ければ history。
    先頭列を DatetimeIndex、数値列に変換し NA 行を落とす。
    """
    if INTRADAY_CSV.exists():
        df = pd.read_csv(INTRADAY_CSV, parse_dates=[0], index_col=0)
    elif HISTORY_CSV.exists():
        df = pd.read_csv(HISTORY_CSV, parse_dates=[0], index_col=0)
    else:
        raise FileNotFoundError("R-BANK9: neither intraday nor history csv found.")

    # すべて数値化（失敗は NaN）
    for c in list(df.columns):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(how="all")

    # タイムゾーン付与/変換（UTC → MARKET_TZ に統一）
    if df.index.tz is None:
        df.index = df.index.tz_localize(timezone.utc).tz_convert(MARKET_TZ)
    else:
        df.index = df.index.tz_convert(MARKET_TZ)

    return df

# =======================
# 1d: ％チャート（始値比）
# =======================
def _gen_1d_percent_png(df: pd.DataFrame, col: str) -> float | None:
    """
    データの「最後に存在するローカル日」を採用し、始値比 (%) の系列を描画。
    戻り値: 終値時点の騰落率(%)。計算不能なら None。
    """
    if df.empty:
        # 空データの場合は空図を保存して終了
        _save_line([], [], OUTDIR / f"{INDEX_KEY}_1d.png",
                   f"{INDEX_KEY.upper()} (1d %)", "Change (%)", FG_TEXT)
        return None

    local_idx = df.index
    last_day_midnight = local_idx[-1].normalize()  # その日の 00:00 in MARKET_TZ
    day_mask = (local_idx.normalize() == last_day_midnight)
    day = df.loc[day_mask, [col]].dropna()

    if len(day) < 2:
        # 観測点が 1 つ以下だと率が出せない
        _save_line([], [], OUTDIR / f"{INDEX_KEY}_1d.png",
                   f"{INDEX_KEY.upper()} (1d %)", "Change (%)", FG_TEXT)
        return None

    open_val = float(day[col].iloc[0])
    series_pct = (day[col] / open_val - 1.0) * 100.0
    last_pct = float(series_pct.iloc[-1])

    # 線色は終値の符号で決定
    color = GREEN if last_pct >= 0 else RED
    _save_line(series_pct.index, series_pct.values,
               OUTDIR / f"{INDEX_KEY}_1d.png",
               f"{INDEX_KEY.upper()} (1d %)", "Change (%)", color)

    return last_pct

# =======================
# 7d/1m/1y: レベル値チャート
# =======================
def _gen_level_pngs(df: pd.DataFrame, col: str) -> None:
    # 7d
    tail_7d = df.tail(7 * 1000)  # 目安：1分足前提で十分長めに
    _save_line(tail_7d.index, tail_7d[col].values,
               OUTDIR / f"{INDEX_KEY}_7d.png",
               f"{INDEX_KEY.upper()} (7d)", "Index / Value", FG_TEXT)

    # 1m（約30日相当）
    tail_1m = df.tail(30 * 1000)
    _save_line(tail_1m.index, tail_1m[col].values,
               OUTDIR / f"{INDEX_KEY}_1m.png",
               f"{INDEX_KEY.upper()} (1m)", "Index / Value", FG_TEXT)

    # 1y（約365日相当）
    tail_1y = df.tail(365 * 1000)
    _save_line(tail_1y.index, tail_1y[col].values,
               OUTDIR / f"{INDEX_KEY}_1y.png",
               f"{INDEX_KEY.upper()} (1y)", "Index / Value", FG_TEXT)

# =======================
# stats / markers
# =======================
def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def _write_stats(pct_1d: float | None) -> None:
    payload = {
        "index_key": INDEX_KEY,
        "pct_1d": None if pct_1d is None else round(pct_1d, 6),
        "scale": "pct",                 # サイト側と合意：百分率
        "updated_at": _now_utc_iso(),
    }
    (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )

def _write_marker(pct_1d: float | None) -> None:
    marker = OUTDIR / f"{INDEX_KEY}_post_intraday.txt"
    if pct_1d is None:
        marker.write_text(f"{INDEX_KEY.upper()} 1d: N/A\n", encoding="utf-8")
    else:
        marker.write_text(f"{INDEX_KEY.upper()} 1d: {pct_1d:+.2f}%\n", encoding="utf-8")

# =======================
# main
# =======================
def main() -> None:
    df = _load_df()
    col = _pick_index_column(df)

    # 1d: ％チャート
    pct_1d = _gen_1d_percent_png(df, col)

    # 7d/1m/1y: レベル値
    _gen_level_pngs(df, col)

    # 出力
    _write_stats(pct_1d)
    _write_marker(pct_1d)

if __name__ == "__main__":
    main()
