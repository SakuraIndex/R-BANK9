#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R-BANK9 charts + stats
- 1d は "レベル" を y 軸に表示（%ではなく算出値そのもの）
- 騰落率は基準=当日の最初の有効値で計算し、stats/post のみ出力
- ダークテーマ / 線色は 基準→終値 の上げ下げで自動切替
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
NEUTRAL = "#a9b2bd"

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

def _save_line(x, y, color, out_png: Path, title: str, y_label: str) -> None:
    fig, ax = plt.subplots()
    _apply(ax, title, y_label)
    ax.plot(x, y, color=color, linewidth=1.6)
    fig.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

# ------------------------
# data loading helpers
# ------------------------
def _to_jst(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """UTC/naive どちらでも JST に揃える（既に tz があれば tz_convert）"""
    if idx.tz is None:
        return idx.tz_localize("UTC").tz_convert("Asia/Tokyo")
    return idx.tz_convert("Asia/Tokyo")

def _pick_index_column(df: pd.DataFrame) -> str:
    """
    優先順位で R-BANK9 の列を決定。無ければ最後の列を使う。
    """
    cand = {
        "rbank9","r_bank9","rbnk9","rbank_9","r_bank_9","r-bank9",
        INDEX_KEY, INDEX_KEY.upper(), "R_BANK9","RBANK9"
    }
    for c in df.columns:
        if c and c.strip().lower() in cand:
            return c
    return df.columns[-1]

def _load_df() -> pd.DataFrame:
    """
    intraday があれば intraday 優先、無ければ history。
    先頭列を DatetimeIndex にして NA を落とす。
    """
    if INTRADAY_CSV.exists():
        df = pd.read_csv(INTRADAY_CSV, parse_dates=[0], index_col=0)
    elif HISTORY_CSV.exists():
        df = pd.read_csv(HISTORY_CSV, parse_dates=[0], index_col=0)
    else:
        raise FileNotFoundError("R-BANK9: neither intraday nor history csv found.")

    # 数値化（非数は NaN）
    for c in list(df.columns):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 全欠損行は落とす
    df = df.dropna(how="all")

    # 時間軸を JST に
    df.index = _to_jst(df.index)
    return df

def _first_valid_of_day(df: pd.DataFrame, col: str):
    """
    当日(JST)の最初の有効値を返す。
    返り値: (ts, value, used_row_idx)
    有効値が見つからなければ None を返す。
    """
    day = df.last('1D')
    if day.empty:
        return None
    ser = day[col].astype(float)
    valid = ser[ser.notna() & np.isfinite(ser.values)]
    if valid.empty:
        return None
    ts = valid.index[0]
    val = float(valid.iloc[0])
    return ts, val, valid.index[0]

# ------------------------
# chart generation (LEVEL on Y for 1d/7d/1m/1y)
# ------------------------
def gen_pngs() -> None:
    df = _load_df()
    col = _pick_index_column(df)

    # 直近ウィンドウ
    d1  = df.last('1D')
    d7  = df.last('7D')
    m1  = df.last('30D')
    y1  = df.last('365D')

    # --- 1d: level chart ---
    # 基準（当日の最初の有効値）で色を決める
    basis_info = _first_valid_of_day(df, col)
    if basis_info is None:
        color = NEUTRAL
        basis_ts = None
        basis_val = None
    else:
        basis_ts, basis_val, _ = basis_info
        # 最新有効値
        last_val = float(d1[col].dropna().iloc[-1]) if not d1.empty and d1[col].dropna().size else np.nan
        color = GREEN if (np.isfinite(last_val) and np.isfinite(basis_val) and last_val >= basis_val) else RED

    # 描画
    if d1.empty or d1[col].dropna().empty:
        _save_line([], [], NEUTRAL, OUTDIR / f"{INDEX_KEY}_1d.png", f"{INDEX_KEY.upper()} (1d level)", "Index (level)")
    else:
        _save_line(d1.index, d1[col], color, OUTDIR / f"{INDEX_KEY}_1d.png", f"{INDEX_KEY.upper()} (1d level)", "Index (level)")

    # --- 長期もレベルで ---
    if not d7.empty and d7[col].dropna().size:
        start7 = float(d7[col].dropna().iloc[0]); last7 = float(d7[col].dropna().iloc[-1])
        color7 = GREEN if last7 >= start7 else RED
        _save_line(d7.index, d7[col], color7, OUTDIR / f"{INDEX_KEY}_7d.png", f"{INDEX_KEY.upper()} (7d level)", "Index (level)")
    if not m1.empty and m1[col].dropna().size:
        startm = float(m1[col].dropna().iloc[0]); lastm = float(m1[col].dropna().iloc[-1])
        colorm = GREEN if lastm >= startm else RED
        _save_line(m1.index, m1[col], colorm, OUTDIR / f"{INDEX_KEY}_1m.png", f"{INDEX_KEY.upper()} (1m level)", "Index (level)")
    if not y1.empty and y1[col].dropna().size:
        starty = float(y1[col].dropna().iloc[0]); lasty = float(y1[col].dropna().iloc[-1])
        colory = GREEN if lasty >= starty else RED
        _save_line(y1.index, y1[col], colory, OUTDIR / f"{INDEX_KEY}_1y.png", f"{INDEX_KEY.upper()} (1y level)", "Index (level)")

    # # 参考：％版も作る場合（必要ならコメントアウト解除）
    # if not d1.empty and d1[col].dropna().size and basis_val not in (None, 0) and np.isfinite(basis_val):
    #     pct = (d1[col] / basis_val - 1.0) * 100.0
    #     color_pct = GREEN if pct.dropna().iloc[-1] >= 0 else RED
    #     _save_line(d1.index, pct, color_pct, OUTDIR / f"{INDEX_KEY}_1d_pct.png", f"{INDEX_KEY.upper()} (1d %)", "Change (%)")

# ------------------------
# stats writer (1d pct) + marker
# ------------------------
def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def write_stats_and_marker() -> None:
    """
    騰落率は「当日の最初の有効値 → 最新有効値」で算出（%）
    先頭行が欠損/ゼロでも安全にスキップ。基準が見つからなければ N/A。
    """
    df = _load_df()
    col = _pick_index_column(df)

    marker_note = ""
    pct = None

    if df.empty or col not in df.columns:
        marker_note = "(no data)"
    else:
        basis = _first_valid_of_day(df, col)
        day = df.last('1D')
        last_series = day[col].dropna() if not day.empty else df[col].dropna()

        if basis is None or last_series.empty:
            marker_note = "(basis missing)"
        else:
            basis_ts, basis_val, _ = basis
            last_val = float(last_series.iloc[-1])
            if not np.isfinite(basis_val) or basis_val == 0:
                marker_note = "(basis invalid→N/A)"
            else:
                pct = (last_val / basis_val - 1.0) * 100.0

                # 先頭行が無効だった場合は注記
                first_row_val = df[col].iloc[0]
                if pd.isna(first_row_val) or not np.isfinite(first_row_val) or float(first_row_val) == 0.0:
                    marker_note = "(basis first-row invalid→used first valid)"

    payload = {
        "index_key": INDEX_KEY,
        "pct_1d": None if pct is None else round(float(pct), 6),
        "scale": "pct",
        "updated_at": _now_utc_iso(),
    }
    (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )

    # human-readable marker
    marker = OUTDIR / f"{INDEX_KEY}_post_intraday.txt"
    if pct is None:
        line = f"{INDEX_KEY.upper()} 1d: N/A"
    else:
        line = f"{INDEX_KEY.upper()} 1d: {pct:+.2f}%"
    if marker_note:
        line += f" {marker_note}"
    marker.write_text(line + "\n", encoding="utf-8")

# ------------------------
# main
# ------------------------
if __name__ == "__main__":
    gen_pngs()
    write_stats_and_marker()
