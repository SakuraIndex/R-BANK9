#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R-BANK9 charts + stats
 - intraday CSV はレベル値（例 1.23, 1.45 ...）
 - 1d の騰落率は「当日セッション（JST 09:00-15:30）の始値→終値」の比で算出
 - 1d グラフは当日セッション内の％推移を描画（色は自動切替）
"""

from pathlib import Path
import json
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
# 基本設定
# ─────────────────────────────────────────────────────────────
INDEX_KEY   = "rbank9"
MARKET_TZ   = ZoneInfo("Asia/Tokyo")
SESSION_S   = time(9, 0)    # 09:00
SESSION_E   = time(15, 30)  # 15:30

OUTDIR = Path("docs/outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

HISTORY_CSV  = OUTDIR / f"{INDEX_KEY}_history.csv"
INTRADAY_CSV = OUTDIR / f"{INDEX_KEY}_intraday.csv"

# ─────────────────────────────────────────────────────────────
# ダークテーマ
# ─────────────────────────────────────────────────────────────
DARK_BG = "#0e0f13"
DARK_AX = "#0b0c10"
FG_TEXT = "#e7ecf1"
GRID    = "#2a2e3a"
GREEN   = "#28e07c"
RED     = "#ff4d4d"

def _style(ax, title: str):
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
    ax.set_ylabel("Change (%)", color=FG_TEXT, fontsize=10)

# ─────────────────────────────────────────────────────────────
# 入出力
# ─────────────────────────────────────────────────────────────
def _load_latest_df() -> pd.DataFrame:
    """ intraday → history の優先で読み込み。index は DatetimeIndex """
    if INTRADAY_CSV.exists():
        df = pd.read_csv(INTRADAY_CSV, parse_dates=[0], index_col=0)
    elif HISTORY_CSV.exists():
        df = pd.read_csv(HISTORY_CSV, parse_dates=[0], index_col=0)
    else:
        raise FileNotFoundError("No intraday/history csv for R-BANK9.")
    # 数値化
    for c in list(df.columns):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # UTC/TZ 付与（既に tz-aware の場合は変換だけ）
    if df.index.tz is None:
        df.index = df.index.tz_localize(timezone.utc)
    else:
        df.index = df.index.tz_convert(timezone.utc)
    # 列名の候補
    cand = {"rbank9", "r_bank9", "rbnk9", "rbank_9", "r_bank_9", "r-bank9", INDEX_KEY, INDEX_KEY.upper(), "R_BANK9", "RBANK9"}
    col = None
    for c in df.columns:
        if c.strip().lower() in cand:
            col = c; break
    if col is None:
        col = df.columns[-1]
    return df.sort_index(), col

def _today_session_window(now_tz: datetime) -> tuple[datetime, datetime]:
    """ ローカル日付の 09:00–15:30 の区間（tz-aware, MARKET_TZ）を返す """
    d = now_tz.date()
    start = datetime.combine(d, SESSION_S, tzinfo=MARKET_TZ)
    end   = datetime.combine(d, SESSION_E, tzinfo=MARKET_TZ)
    if end <= start:
        end += timedelta(days=1)
    return start, end

def _slice_today_session(df_utc: pd.DataFrame) -> pd.DataFrame:
    """ UTC index → MARKET_TZ へ変換後、当日セッションで切り出し """
    df_local = df_utc.copy()
    df_local.index = df_local.index.tz_convert(MARKET_TZ)
    now_local = datetime.now(MARKET_TZ)
    s, e = _today_session_window(now_local)
    return df_local[(df_local.index >= s) & (df_local.index <= e)]

# ─────────────────────────────────────────────────────────────
# 1d グラフと騰落率
# ─────────────────────────────────────────────────────────────
def _plot_1d_pct(df_local: pd.DataFrame, col: str, outpath: Path):
    """ 当日セッション内の％推移を描画。色は始値/終値で自動切替 """
    # 有効値だけ
    ser = df_local[col].dropna()
    fig, ax = plt.subplots()
    _style(ax, f"{INDEX_KEY.upper()} (1d %)")

    if len(ser) < 2:
        # データが足りないときは 0% の水平線を描く
        ax.plot(df_local.index, [0.0] * len(df_local.index), color=FG_TEXT, linewidth=1.5)
        fig.savefig(outpath, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return None  # pct_1d は別で計算する

    first = ser.iloc[0]
    last  = ser.iloc[-1]
    if first == 0 or pd.isna(first) or pd.isna(last):
        color = FG_TEXT
        pct_series = pd.Series([0.0]*len(ser), index=ser.index)
        pct_val = None
    else:
        pct_series = (ser / first - 1.0) * 100.0
        pct_val = (last / first - 1.0) * 100.0
        color = GREEN if pct_val >= 0 else RED

    ax.plot(pct_series.index, pct_series.values, color=color, linewidth=1.6)
    fig.savefig(outpath, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return pct_val

def _write_stats_and_marker(pct_1d):
    payload = {
        "index_key": INDEX_KEY,
        "pct_1d": None if pct_1d is None else float(round(pct_1d, 6)),
        "scale": "pct",
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    txt = f"{INDEX_KEY.upper()} 1d: " + ("N/A" if pct_1d is None else f"{pct_1d:+.2f}%") + "\n"
    (OUTDIR / f"{INDEX_KEY}_post_intraday.txt").write_text(txt, encoding="utf-8")

# 参考：7d/1m/1y は見た目の互換性維持のため「レベル値の折れ線」のまま出力
def _simple_level_plot(df_utc: pd.DataFrame, col: str, tail_points: int, title: str, outpath: Path):
    df = df_utc.tail(tail_points).copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize(timezone.utc)
    df.index = df.index.tz_convert(MARKET_TZ)
    ser = df[col]
    color = FG_TEXT
    if len(ser.dropna()) >= 2:
        first, last = ser.dropna().iloc[0], ser.dropna().iloc[-1]
        if pd.notna(first) and pd.notna(last):
            color = GREEN if last >= first else RED

    fig, ax = plt.subplots()
    # ラベルは「Index / Value」のかわりに実態を示す
    ax.set_ylabel("Index level", color=FG_TEXT, fontsize=10)
    _style(ax, title)
    ax.plot(df.index, ser, color=color, linewidth=1.2)
    fig.savefig(outpath, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────
def main():
    df_utc, col = _load_latest_df()

    # 当日セッションを JST で切り出し
    today_local = _slice_today_session(df_utc)

    # 1D（％）を描画 & 当日 pct を計算
    pct_1d = _plot_1d_pct(
        df_local=today_local,
        col=col,
        outpath=OUTDIR / f"{INDEX_KEY}_1d.png",
    )

    # 7d/1m/1y は従来どおりレベル値の推移（見た目だけ色切替）
    _simple_level_plot(df_utc, col, tail_points=7*24*12,  title=f"{INDEX_KEY.upper()} (7d)", outpath=OUTDIR / f"{INDEX_KEY}_7d.png")
    _simple_level_plot(df_utc, col, tail_points=30*24*12, title=f"{INDEX_KEY.upper()} (1m)", outpath=OUTDIR / f"{INDEX_KEY}_1m.png")
    _simple_level_plot(df_utc, col, tail_points=365*24*12,title=f"{INDEX_KEY.upper()} (1y)", outpath=OUTDIR / f"{INDEX_KEY}_1y.png")

    # 統計とマーカーを常に書き直す
    _write_stats_and_marker(pct_1d)

if __name__ == "__main__":
    main()
