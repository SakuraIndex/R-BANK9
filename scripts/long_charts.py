#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R-BANK9 charts + stats (完全版)
 - 1d = セッション内％推移（始値基準）
 - 7d / 1m / 1y = ウィンドウ先頭基準％推移
 - タイムゾーン安全 (naive/UTC/JST/tz-aware すべて対応)
 - エラー耐性: データ不足時は水平線＆pct_1d=None
 - ダークテーマ、線色は上昇=緑、下落=赤
"""

from pathlib import Path
import os
import json
from datetime import datetime, time, timedelta, timezone
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================
# 設定
# =========================
INDEX_KEY = "rbank9"
OUTDIR = Path("docs/outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

HISTORY_CSV  = OUTDIR / f"{INDEX_KEY}_history.csv"
INTRADAY_CSV = OUTDIR / f"{INDEX_KEY}_intraday.csv"

MARKET_TZ     = os.environ.get("MARKET_TZ", "Asia/Tokyo")
SESSION_START = os.environ.get("SESSION_START", "09:00")
SESSION_END   = os.environ.get("SESSION_END", "15:30")


# =========================
# グラフスタイル
# =========================
DARK_BG = "#0e0f13"
DARK_AX = "#0b0c10"
FG_TEXT = "#e7ecf1"
GRID    = "#2a2e3a"
GREEN   = "#28e07c"
RED     = "#ff4d4d"


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


def _save_line(x, y, out_png: Path, title: str, y_label: str) -> None:
    if len(y) >= 2 and np.isfinite(y.iloc[0]) and np.isfinite(y.iloc[-1]):
        color = GREEN if y.iloc[-1] >= y.iloc[0] else RED
    else:
        color = FG_TEXT

    fig, ax = plt.subplots()
    _apply(ax, title, y_label)
    ax.plot(x, y, color=color, linewidth=1.6)
    fig.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# =========================
# 読み込み / タイムゾーン処理
# =========================
def _load_df() -> pd.DataFrame:
    if INTRADAY_CSV.exists():
        df = pd.read_csv(INTRADAY_CSV, parse_dates=[0], index_col=0)
    elif HISTORY_CSV.exists():
        df = pd.read_csv(HISTORY_CSV, parse_dates=[0], index_col=0)
    else:
        raise FileNotFoundError("No data CSV found.")

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(how="all")
    return _to_jst(df)


def _detect_tz_for_naive(df: pd.DataFrame) -> str:
    """naive index が UTC か JST かを簡易判定（大半が9〜15時ならJST）"""
    h = df.index.hour
    return "JST" if ((h >= 9) & (h <= 15)).mean() > 0.5 else "UTC"


def _to_jst(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    idx = df.index
    if idx.tz is None:
        tztype = _detect_tz_for_naive(df)
        if tztype == "UTC":
            idx = idx.tz_localize("UTC").tz_convert(MARKET_TZ)
        else:
            idx = idx.tz_localize(MARKET_TZ)
    else:
        idx = idx.tz_convert(MARKET_TZ)

    df = df.copy()
    df.index = idx
    return df


def _pick_index_column(df: pd.DataFrame) -> str:
    cand = {
        "rbank9", "r_bank9", "rbnk9", "rbank_9", "r_bank_9", "r-bank9",
        INDEX_KEY, INDEX_KEY.upper(), "R_BANK9", "RBANK9"
    }
    for c in df.columns:
        if c.strip().lower() in cand:
            return c
    return df.columns[-1]


# =========================
# 抽出＆％変換
# =========================
def _today_session_window():
    tz = pd.Timestamp.now(tz=MARKET_TZ).tz
    now = pd.Timestamp.now(tz)
    sh, sm = map(int, SESSION_START.split(":"))
    eh, em = map(int, SESSION_END.split(":"))
    start = pd.Timestamp(year=now.year, month=now.month, day=now.day, hour=sh, minute=sm, tz=tz)
    end = pd.Timestamp(year=now.year, month=now.month, day=now.day, hour=eh, minute=em, tz=tz)
    if end <= start:
        end = end + pd.Timedelta(days=1)
    return start.to_pydatetime(), end.to_pydatetime()


def _slice(df: pd.DataFrame, start, end) -> pd.DataFrame:
    return df[(df.index >= start) & (df.index <= end)]


def _to_pct_series(series: pd.Series, base: float) -> pd.Series:
    if not np.isfinite(base) or base == 0:
        return pd.Series(dtype=float, index=series.index)
    return (series / base - 1.0) * 100.0


# =========================
# 生成メイン
# =========================
def gen_pngs_and_stats() -> None:
    df = _load_df()
    col = _pick_index_column(df)

    # ---- 1d ----
    s, e = _today_session_window()
    day = _slice(df, s, e)
    day = day[[col]].dropna()

    if len(day) >= 2:
        base = float(day[col].iloc[0])
        pct1d = _to_pct_series(day[col], base)
        _save_line(pct1d.index, pct1d, OUTDIR / f"{INDEX_KEY}_1d.png", f"{INDEX_KEY.upper()} (1d %)", "Change (%)")
        last_pct = float(pct1d.iloc[-1])
    else:
        zero = pd.Series([0.0, 0.0],
                         index=pd.DatetimeIndex([
                             pd.Timestamp.now(tz=MARKET_TZ) - pd.Timedelta(hours=1),
                             pd.Timestamp.now(tz=MARKET_TZ)
                         ]))
        _save_line(zero.index, zero, OUTDIR / f"{INDEX_KEY}_1d.png", f"{INDEX_KEY.upper()} (1d %)", "Change (%)")
        last_pct = None

    # ---- 7d / 1m / 1y ----
    def _window_pct(days: int, title: str, out: Path):
        w = df[[col]].last(f"{days}D").dropna()
        if len(w) >= 2:
            base = float(w[col].iloc[0])
            pct = _to_pct_series(w[col], base)
            _save_line(pct.index, pct, out, f"{INDEX_KEY.upper()} ({title} %)", "Change (%)")
        else:
            zero = pd.Series([0.0, 0.0],
                             index=pd.DatetimeIndex([
                                 pd.Timestamp.now(tz=MARKET_TZ) - pd.Timedelta(days=days - 1),
                                 pd.Timestamp.now(tz=MARKET_TZ)
                             ]))
            _save_line(zero.index, zero, out, f"{INDEX_KEY.upper()} ({title} %)", "Change (%)")

    _window_pct(7, "7d", OUTDIR / f"{INDEX_KEY}_7d.png")
    _window_pct(30, "1m", OUTDIR / f"{INDEX_KEY}_1m.png")
    _window_pct(365, "1y", OUTDIR / f"{INDEX_KEY}_1y.png")

    # ---- stats & post file ----
    payload = {
        "index_key": INDEX_KEY,
        "pct_1d": None if last_pct is None or not np.isfinite(last_pct) else round(float(last_pct), 6),
        "scale": "pct",
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    marker = OUTDIR / f"{INDEX_KEY}_post_intraday.txt"
    if payload["pct_1d"] is None:
        marker.write_text(f"{INDEX_KEY.upper()} 1d: N/A\n", encoding="utf-8")
    else:
        marker.write_text(f"{INDEX_KEY.upper()} 1d: {payload['pct_1d']:+.2f}%\n", encoding="utf-8")


# =========================
# main
# =========================
if __name__ == "__main__":
    gen_pngs_and_stats()
