#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R-BANK9 charts + stats
- 1d は「%変化」を描画（サイトの騰落率と一致）
- セッション内の「最初の非NaNかつ非0」を始値として採用（0.0 を除外）
- タイムゾーンは CSV の tz を尊重し、JST セッション(09:00-15:30)で切り出し
- 線色は終値>=始値で緑、そうでなければ赤
"""

from __future__ import annotations
from pathlib import Path
import json
from datetime import datetime, date, time, timedelta, timezone
from zoneinfo import ZoneInfo

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

MARKET_TZ = ZoneInfo("Asia/Tokyo")
SESSION_START = time(9, 0)   # 09:00 JST
SESSION_END   = time(15, 30) # 15:30 JST

# ------------------------
# plotting style (dark)
# ------------------------
DARK_BG = "#0e0f13"
DARK_AX = "#0b0c10"
FG_TEXT = "#e7ecf1"
GRID    = "#2a2e3a"
GREEN   = "#28e07c"   # 上昇
RED     = "#ff4d4d"   # 下落
GREY    = "#cccccc"   # データなし

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

def _save_line(x, y, out_png: Path, title: str, y_label: str, color: str) -> None:
    fig, ax = plt.subplots()
    _apply(ax, title, y_label)
    if len(x) > 0:
        ax.plot(x, y, color=color, linewidth=1.6)
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
        if c.strip().lower() in cand_names:
            return c
    return df.columns[-1]

def _read_any() -> pd.DataFrame:
    """
    intraday があれば intraday 優先、無ければ history。
    先頭列（無名でも可）を DatetimeIndex として読む。
    tz 付きなら尊重、tz 無しなら UTC として扱い(JSTへ変換時に tz_localize)。
    """
    if INTRADAY_CSV.exists():
        src = INTRADAY_CSV
    elif HISTORY_CSV.exists():
        src = HISTORY_CSV
    else:
        raise FileNotFoundError("R-BANK9: neither intraday nor history csv found.")

    # 先頭列名が空でも parse_dates=[0], index_col=0 でOK
    df = pd.read_csv(src, parse_dates=[0], index_col=0)

    # 数値化（非数は NaN）
    for c in list(df.columns):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # タイムゾーン調整
    idx = df.index
    if getattr(idx, "tz", None) is None:
        # naive -> UTC と仮定
        df.index = pd.DatetimeIndex(df.index.tz_localize(timezone.utc))
    else:
        # tz-aware -> そのまま
        df.index = pd.DatetimeIndex(df.index)

    return df.sort_index()

def _today_session_window() -> tuple[datetime, datetime]:
    """
    本日JSTの 09:00〜15:30 の時間窓（tz-aware, JST）
    """
    now_jst = datetime.now(MARKET_TZ)
    d: date = now_jst.date()
    start = datetime(d.year, d.month, d.day, SESSION_START.hour, SESSION_START.minute, tzinfo=MARKET_TZ)
    end   = datetime(d.year, d.month, d.day, SESSION_END.hour,   SESSION_END.minute,   tzinfo=MARKET_TZ)
    # 念のため end < start の場合は翌日にずらす（今回は不要想定だが安全策）
    if end <= start:
        end += timedelta(days=1)
    return start, end

def _slice_today_session(df: pd.DataFrame) -> pd.DataFrame:
    """
    データを JST の当日セッション窓で切り出して返す（UTCをJSTへ変換してからスライス）
    """
    df_jst = df.tz_convert(MARKET_TZ)
    start, end = _today_session_window()
    return df_jst.loc[(df_jst.index >= start) & (df_jst.index <= end)].copy()

# ------------------------
# 1d % series & stats
# ------------------------
def _build_1d_change_series(df_sess: pd.DataFrame, col: str) -> pd.Series:
    """
    セッション中の「最初の非NaNかつ非0」を基準（ベース）にして %変化を出す。
    戻り値: パーセント（%）の Series（JST 時刻が index）
    """
    y = df_sess[col].dropna()
    # 非0の最初の値を探す
    first_valid = None
    for v in y.values:
        if pd.notna(v) and v != 0:
            first_valid = float(v)
            break

    if first_valid is None:
        # データなし
        return pd.Series(dtype="float64", index=df_sess.index.tz_convert(MARKET_TZ))

    # %変化 = (値 / 始値 - 1) * 100
    pct = (y / first_valid - 1.0) * 100.0
    # 元の時刻全体に合わせた reindex（欠損は前方埋めせず、そのままNaN）
    pct = pct.reindex(df_sess.index)
    return pct

def _compute_close_pct(pct_series: pd.Series) -> float | None:
    """終値の % を返す（有効な最後の数値がなければ None）"""
    if pct_series is None or pct_series.empty:
        return None
    last = pct_series.dropna().tail(1)
    if last.empty:
        return None
    return float(last.iloc[0])

# ------------------------
# chart generation
# ------------------------
def gen_pngs_and_stats() -> None:
    df_all = _read_any()
    col = _pick_index_column(df_all)

    # --- 1d: % のグラフ（サイトで使用） ---
    sess = _slice_today_session(df_all)
    pct_1d_series = _build_1d_change_series(sess, col)
    last_pct = _compute_close_pct(pct_1d_series)

    # 線色（データなし:グレー / 0%以上:緑 / <0%:赤）
    if last_pct is None:
        color_1d = GREY
    else:
        color_1d = GREEN if last_pct >= 0 else RED

    _save_line(
        pct_1d_series.index, pct_1d_series.values,
        OUTDIR / f"{INDEX_KEY}_1d.png",
        f"{INDEX_KEY.upper()} (1d %)",
        "Change (%)",
        color_1d
    )

    # --- 7d/1m/1y は従来どおり“レベル”をプロット（参考用） ---
    def _save_level(window_count: int, label: str, fname: str):
        tail = df_all.tail(window_count)
        y = tail[col].dropna()
        if y.empty:
            _save_line(tail.index, [], OUTDIR / fname, f"{INDEX_KEY.upper()} ({label})", "Index / Value", GREY)
            return
        first, last = y.iloc[0], y.iloc[-1]
        color = GREEN if (pd.notna(first) and pd.notna(last) and last >= first) else RED
        _save_line(y.index, y.values, OUTDIR / fname, f"{INDEX_KEY.upper()} ({label})", "Index / Value", color)

    _save_level(7 * 1000,  "7d",  f"{INDEX_KEY}_7d.png")
    _save_level(30 * 1000, "1m",  f"{INDEX_KEY}_1m.png")
    _save_level(365 * 1000,"1y",  f"{INDEX_KEY}_1y.png")

    # --- stats.json & human marker ---
    payload = {
        "index_key": INDEX_KEY,
        "pct_1d": None if last_pct is None else round(last_pct, 6),
        "scale": "pct",  # サイト側は百分率で解釈
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )

    marker = OUTDIR / f"{INDEX_KEY}_post_intraday.txt"
    if last_pct is None:
        marker.write_text(f"{INDEX_KEY.upper()} 1d: N/A\n", encoding="utf-8")
    else:
        marker.write_text(f"{INDEX_KEY.upper()} 1d: {last_pct:+.2f}%\n", encoding="utf-8")

# ------------------------
# main
# ------------------------
if __name__ == "__main__":
    gen_pngs_and_stats()
