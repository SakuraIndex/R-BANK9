#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R-BANK9 charts + stats
 - 1d は「％」チャートに統一（JSTの場中のみを対象）
 - ロバスト基準値（最初の15分の中央値）で日中のスパイクに耐性
 - 7d/1m/1y は従来どおりレベル値（参考）
 - ダークテーマ / 線色は終値の符号で自動切替
 - stats.json と post_intraday.txt を出力
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, time, timedelta, timezone
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------
# settings / paths
# ------------------------
INDEX_KEY   = "rbank9"
OUTDIR      = Path("docs/outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

HISTORY_CSV  = OUTDIR / f"{INDEX_KEY}_history.csv"
INTRADAY_CSV = OUTDIR / f"{INDEX_KEY}_intraday.csv"

# 市場（JST）営業時間
MARKET_TZ   = "Asia/Tokyo"
SESSION_S   = time(9, 0)   # 09:00
SESSION_E   = time(15, 30) # 15:30
BASE_WINDOW_MIN = 15       # 基準値に使う最初の15分の中央値

# ダークテーマ
DARK_BG = "#0e0f13"
DARK_AX = "#0b0c10"
FG_TEXT = "#e7ecf1"
GRID    = "#2a2e3a"
GREEN   = "#28e07c"
RED     = "#ff4d4d"

# ------------------------
# helpers
# ------------------------
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

def _pick_index_column(df: pd.DataFrame) -> str:
    """
    R-BANK9 本体列を推定。候補が無ければ最後の列。
    """
    cand = {INDEX_KEY, INDEX_KEY.upper(), "r_bank9", "r-bank9", "R_BANK9", "RBANK9"}
    for c in df.columns:
        if c.strip().lower() in cand:
            return c
    return df.columns[-1]

def _load_df() -> pd.DataFrame:
    """
    intraday があれば優先、無ければ history。
    先頭列を時刻に、数値に変換。
    """
    if INTRADAY_CSV.exists():
        df = pd.read_csv(INTRADAY_CSV, parse_dates=[0], index_col=0)
    elif HISTORY_CSV.exists():
        df = pd.read_csv(HISTORY_CSV, parse_dates=[0], index_col=0)
    else:
        raise FileNotFoundError("No intraday/history csv found.")
    # 数値化
    for c in list(df.columns):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(how="all")
    return df

def _to_jst_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if idx.tz is None:
        # CSVがオフセット付きなら pandas がtzawareで読む。無ければUTC想定。
        idx = idx.tz_localize(timezone.utc)
    return idx.tz_convert(MARKET_TZ)

def _last_session_slice(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    直近（JST）の営業日セッション [09:00, 15:30] のみを抽出。
    当日のデータが無ければ最後に観測された日付を対象。
    """
    if df.empty: 
        return df
    df = df.copy()
    df.index = _to_jst_index(df.index)

    # 直近の観測日の JST 日付
    last_day = df.index[-1].date()

    # その日のセッションウィンドウ
    start = datetime.combine(last_day, SESSION_S).astimezone(df.index.tz)
    end   = datetime.combine(last_day, SESSION_E).astimezone(df.index.tz)
    if end <= start:
        end += timedelta(days=1)

    mask = (df.index >= start) & (df.index <= end)
    day_df = df.loc[mask]

    # もし空なら、最後の観測日から逆順で近い日を探す（休日・CSV欠落対策）
    if day_df.empty:
        for dshift in range(1, 6):  # 最大5営業日分遡る簡易対策
            d = last_day - timedelta(days=dshift)
            start = datetime.combine(d, SESSION_S).astimezone(df.index.tz)
            end   = datetime.combine(d, SESSION_E).astimezone(df.index.tz)
            if end <= start:
                end += timedelta(days=1)
            mask = (df.index >= start) & (df.index <= end)
            day_df = df.loc[mask]
            if not day_df.empty:
                break
    return day_df

def _robust_base(values: pd.Series) -> float | None:
    """
    最初の BASE_WINDOW_MIN 分の中央値を基準値にする。
    NaN/0/負値は無視。十分な点が無ければ None。
    """
    if values.empty:
        return None
    # 最初の時間帯
    t0 = values.index[0]
    t1 = t0 + timedelta(minutes=BASE_WINDOW_MIN)
    v = values[(values.index >= t0) & (values.index <= t1)]
    v = pd.to_numeric(v, errors="coerce").dropna()
    v = v[v > 0]
    if len(v) < 2:
        return None
    return float(np.median(v.values))

def _save_line(x: pd.DatetimeIndex, y: pd.Series, out_png: Path, title: str, ylabel: str, color: str) -> None:
    fig, ax = plt.subplots()
    _apply(ax, title, ylabel)
    ax.plot(x, y, color=color, linewidth=1.6)
    fig.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

# ------------------------
# main generators
# ------------------------
def gen_pngs_and_stats() -> None:
    df = _load_df()
    if df.empty:
        # 空ならN/Aを書いてリターン
        _write_stats(None, None, None)
        return

    col = _pick_index_column(df)

    # --- 1d: %チャート（JST場中）
    day = _last_session_slice(df[[col]], col)
    pct_last = None
    base_used = None
    last_val  = None

    if not day.empty:
        base_used = _robust_base(day[col])
        last_val  = float(day[col].iloc[-1]) if pd.notna(day[col].iloc[-1]) else None
        if base_used and last_val:
            pct_series = (day[col] / base_used - 1.0) * 100.0
            pct_last = float(pct_series.iloc[-1])
            color = GREEN if pct_last >= 0 else RED
            _save_line(
                x=day.index, y=pct_series,
                out_png=OUTDIR / f"{INDEX_KEY}_1d.png",
                title=f"{INDEX_KEY.upper()} (1d %)",
                ylabel="Change (%)",
                color=color
            )
        else:
            # 基準値が取れなかった場合でも、空のフレームで黒背景PNGだけ更新
            _save_line(day.index, day[col].pipe(lambda s: s*0), OUTDIR / f"{INDEX_KEY}_1d.png",
                       f"{INDEX_KEY.upper()} (1d %)", "Change (%)", FG_TEXT)
    else:
        # セッション抽出が空の場合の保険
        _save_line(df.index, df[col].pipe(lambda s: s*0), OUTDIR / f"{INDEX_KEY}_1d.png",
                   f"{INDEX_KEY.upper()} (1d %)", "Change (%)", FG_TEXT)

    # --- 参考: レベル値の7d/1m/1y（従来の見た目）
    #   ※ここは「値」そのものなので色は固定の中立色でOK
    def _plot_level(slice_df: pd.DataFrame, horizon: str):
        if slice_df.empty:
            return
        _save_line(
            x=slice_df.index, y=slice_df[col],
            out_png=OUTDIR / f"{INDEX_KEY}_{horizon}.png",
            title=f"{INDEX_KEY.upper()} ({horizon})",
            ylabel="Index / Value",
            color=FG_TEXT
        )

    # tailサイズは経験則（必要なら調整）
    df_lv = df.copy()
    df_lv.index = _to_jst_index(df_lv.index)
    _plot_level(df_lv.tail(7*1000),  "7d")
    _plot_level(df_lv.tail(30*1000), "1m")
    _plot_level(df_lv.tail(365*1000),"1y")

    # --- stats & marker
    _write_stats(pct_last, base_used, last_val)

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00","Z")

def _write_stats(pct_last: float | None, base: float | None, last_val: float | None) -> None:
    payload = {
        "index_key": INDEX_KEY,
        "pct_1d": None if pct_last is None or np.isnan(pct_last) else round(float(pct_last), 6),
        "scale": "pct",
        "updated_at": _now_utc_iso(),
    }
    (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8"
    )

    # human-readable marker（基準値と終値も出す）
    marker = OUTDIR / f"{INDEX_KEY}_post_intraday.txt"
    if pct_last is None or np.isnan(pct_last):
        marker.write_text(f"{INDEX_KEY.upper()} 1d: N/A\n", encoding="utf-8")
    else:
        base_s = "N/A" if base is None else f"{base:.4f}"
        last_s = "N/A" if last_val is None else f"{last_val:.4f}"
        marker.write_text(
            f"{INDEX_KEY.upper()} 1d: {pct_last:+.2f}% (base={base_s}, last={last_s})\n",
            encoding="utf-8"
        )

# ------------------------
# entry
# ------------------------
if __name__ == "__main__":
    gen_pngs_and_stats()
