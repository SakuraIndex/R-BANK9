#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate long-term charts (1d/7d/1m/1y) for R-BANK9.
- 日本株 (JST 9:00-15:30) のセッションで1dを切り出し
- intraday は (1) time/value[/volume] 形式 も (2) 銘柄列が横並びの板状形式 も対応
  → 板状形式は数値列を等加重平均して value を作成
- 列名は大文字/小文字/日本語/Unnamed も吸収、時間列が無い場合は先頭列が時刻っぽければ採用
- 1d は 始値<終値: 青緑 / 始値>終値: 赤 / 同値: グレー
出力先: docs/outputs/rbank9_{1d|7d|1m|1y}.png
"""

import os
import re
from datetime import timedelta
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ============================================================
#  基本設定
# ============================================================

INDEX_KEY = os.environ.get("INDEX_KEY", "rbank9").lower()
OUTPUT_DIR = "docs/outputs"

# 色
COLOR_PRICE_DEFAULT = "#ff99cc"  # 長期線
COLOR_VOLUME = "#7f8ca6"
COLOR_UP = "#00C2A0"   # 陽線
COLOR_DOWN = "#FF4C4C" # 陰線
COLOR_EQUAL = "#AAAAAA"

plt.rcParams.update({
    "font.family": "Noto Sans CJK JP",
    "figure.facecolor": "#0b0f1a",
    "axes.facecolor": "#0b0f1a",
    "axes.edgecolor": "#27314a",
    "axes.labelcolor": "#e5ecff",
    "xtick.color": "#b8c2e0",
    "ytick.color": "#b8c2e0",
    "grid.color": "#27314a",
})

def log(msg: str):
    print(f"[long_charts] {msg}")

# ============================================================
#  市場セッション (R-BANK9 は日本株)
# ============================================================

MARKET = dict(
    RAW_TZ_INTRADAY="Asia/Tokyo",
    RAW_TZ_HISTORY="Asia/Tokyo",
    DISPLAY_TZ="Asia/Tokyo",
    SESSION_TZ="Asia/Tokyo",
    SESSION_START=(9, 0),   # 9:00
    SESSION_END=(15, 30),   # 15:30
)

# ============================================================
#  入出力ヘルパ
# ============================================================

def _first(paths: List[str]) -> Optional[str]:
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def find_intraday(base: str, key: str) -> Optional[str]:
    return _first([
        f"{base}/{key}_intraday.csv",
        f"{base}/{key}_intraday.txt",
    ])

def find_history(base: str, key: str) -> Optional[str]:
    return _first([
        f"{base}/{key}_history.csv",
        f"{base}/{key}_history.txt",
    ])

# ============================================================
#  時刻/列検出
# ============================================================

TIME_CANDIDATES = [
    "datetime", "time", "timestamp", "date",
    "時刻", "日時", "日付", "時間",
    "unnamed: 0", "Unnamed: 0"
]

def _looks_like_datetime_series(s: pd.Series) -> bool:
    try:
        parsed = pd.to_datetime(s, errors="coerce", utc=True)
        return parsed.notna().mean() > 0.7
    except Exception:
        return False

def parse_time_any(x, raw_tz: str, display_tz: str):
    if pd.isna(x):
        return pd.NaT
    s = str(x).strip()

    # UNIX 13桁(ms) / 10桁(s)
    if re.fullmatch(r"\d{13}", s):
        return pd.Timestamp(int(s), unit="ms", tz="UTC").tz_convert(display_tz)
    if re.fullmatch(r"\d{10}", s):
        return pd.Timestamp(int(s), unit="s", tz="UTC").tz_convert(display_tz)

    try:
        t = pd.to_datetime(s, errors="coerce")
        if pd.isna(t):
            return pd.NaT
        if t.tzinfo is None:
            t = t.tz_localize(raw_tz)
        return t.tz_convert(display_tz)
    except Exception:
        return pd.NaT

def pick_time_col(cols_lower: List[str]) -> Optional[str]:
    for name in TIME_CANDIDATES:
        lower = name.lower()
        if lower in cols_lower:
            return cols_lower[cols_lower.index(lower)]
    # fuzzy（"date"/"time" を含む最初）
    fuzzy = [c for c in cols_lower if ("time" in c) or ("date" in c)]
    return fuzzy[0] if fuzzy else None

def pick_value_col(df: pd.DataFrame) -> Optional[str]:
    # よくある候補名
    candidates = ["close", "value", "index", "価格", "終値", "終値(円)", "終値_指数", "score"]
    cols_lower = [c.lower() for c in df.columns]
    for k in candidates:
        if k.lower() in cols_lower:
            return df.columns[cols_lower.index(k.lower())]
    # 数値列が1本しかないならそれを採用
    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(nums) == 1:
        return nums[0]
    return None

def pick_volume_col(df: pd.DataFrame) -> Optional[str]:
    candidates = ["volume", "vol", "出来高", "出来高(株)"]
    cols_lower = [c.lower() for c in df.columns]
    for k in candidates:
        if k.lower() in cols_lower:
            return df.columns[cols_lower.index(k.lower())]
    return None

# ============================================================
#  CSV 読み込み（time/value/volume の標準化）
# ============================================================

def read_any(path: Optional[str], raw_tz: str, display_tz: str) -> pd.DataFrame:
    """
    下記どちらの形式にも対応して標準化 DataFrame を返す:
      (1) time/value[/volume] 形式
      (2) 銘柄が横持ちの板状形式（時間列 + 複数の銘柄列）→ 数値列の等加重平均で value を生成
    """
    if not path:
        return pd.DataFrame(columns=["time", "value", "volume"])

    df = pd.read_csv(path)

    # 列名正規化
    raw_cols = list(df.columns)
    cols_lower = [str(c).strip().lower() for c in raw_cols]
    df.columns = cols_lower

    # ---- 時刻列を特定
    tcol = pick_time_col(cols_lower)
    if tcol is None:
        # 先頭列が時刻っぽければ採用
        first_col = df.columns[0]
        if _looks_like_datetime_series(df[first_col]):
            tcol = first_col
        else:
            raise KeyError(f"No time-like column found. columns={list(df.columns)}")

    # ---- (1) time/value[/volume] 形式の検出
    vcol = pick_value_col(df)
    volcol = pick_volume_col(df)
    if vcol is not None and (volcol is None or volcol in df.columns):
        out = pd.DataFrame()
        out["time"] = df[tcol].apply(lambda x: parse_time_any(x, raw_tz, display_tz))
        out["value"] = pd.to_numeric(df[vcol], errors="coerce")
        out["volume"] = pd.to_numeric(df[volcol], errors="coerce") if (volcol and volcol in df.columns) else 0
        out = out.dropna(subset=["time", "value"]).sort_values("time").reset_index(drop=True)
        return out

    # ---- (2) 盤面形式：時刻列以外の数値列を平均
    num_cols: List[str] = []
    for c in df.columns:
        if c == tcol:
            continue
        as_num = pd.to_numeric(df[c], errors="coerce")
        if as_num.notna().sum() > 0:
            num_cols.append(c)

    if not num_cols:
        return pd.DataFrame(columns=["time", "value", "volume"])

    out = pd.DataFrame()
    out["time"] = df[tcol].apply(lambda x: parse_time_any(x, raw_tz, display_tz))
    vals_df = df[num_cols].apply(lambda s: pd.to_numeric(s, errors="coerce"))
    out["value"] = vals_df.mean(axis=1)
    out["volume"] = 0
    out = out.dropna(subset=["time", "value"]).sort_values("time").reset_index(drop=True)
    return out

# ============================================================
#  集計
# ============================================================

def to_daily(df: pd.DataFrame, display_tz: str) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.copy()
    d["date"] = d["time"].dt.tz_convert(display_tz).dt.date
    g = d.groupby("date", as_index=False).agg({"value": "last", "volume": "sum"})
    g["time"] = pd.to_datetime(g["date"]).dt.tz_localize(display_tz)
    return g[["time", "value", "volume"]]

# ============================================================
#  グラフ補助
# ============================================================

def format_time_axis(ax, mode: str, tz: str):
    if mode == "1d":
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1, tz=tz))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tz))
    else:
        loc = mdates.AutoDateLocator(minticks=3, maxticks=6, tz=tz)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))

def apply_y_padding(ax, series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        ax.set_ylim(0, 1)
        return
    lo, hi = s.min(), s.max()
    pad = (hi - lo) * 0.08 if hi != lo else max(abs(lo) * 0.02, 0.5)
    ax.set_ylim(lo - pad, hi + pad)

def jst_session_frame(base_ts_jst: pd.Timestamp, start_hm, end_hm, tz: str):
    day = base_ts_jst.tz_convert(tz).date()
    start = pd.Timestamp(day.year, day.month, day.day, start_hm[0], start_hm[1], tz=tz)
    end = pd.Timestamp(day.year, day.month, day.day, end_hm[0], end_hm[1], tz=tz)
    return start, end

# ============================================================
#  描画
# ============================================================

def plot_df(df: pd.DataFrame, index_key: str, label: str, mode: str, tz: str, frame=None):
    fig, ax1 = plt.subplots(figsize=(9.5, 4.8))
    ax1.grid(True, alpha=0.3)

    if df is None or df.empty:
        # 空でも軸は見せる
        ax1.text(0.5, 0.5, "No data", transform=ax1.transAxes, ha="center", va="center", color="#b8c2e0", fontsize=18)
        ax1.set_title(f"{index_key.upper()} ({label})", color="#ffb6c1")
        ax1.set_xlabel("Time" if mode == "1d" else "Date")
        ax1.set_ylabel("Index Value")
        format_time_axis(ax1, mode, tz)
        if frame is not None:
            ax1.set_xlim(frame)
        plt.tight_layout()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.savefig(f"{OUTPUT_DIR}/{index_key}_{label}.png", dpi=180)
        plt.close()
        return

    # 1d の色分け
    if mode == "1d":
        open_price = df["value"].iloc[0]
        close_price = df["value"].iloc[-1]
        if close_price > open_price:
            color_line = COLOR_UP
        elif close_price < open_price:
            color_line = COLOR_DOWN
        else:
            color_line = COLOR_EQUAL
        lw = 2.2
    else:
        color_line = COLOR_PRICE_DEFAULT
        lw = 1.8

    # 出来高（あれば）
    if "volume" in df.columns and np.nan_to_num(df["volume"].values, nan=0.0).sum() > 0:
        ax2 = ax1.twinx()
        ax2.bar(df["time"], df["volume"], width=0.9 if mode == "1d" else 0.8,
                color=COLOR_VOLUME, alpha=0.35, zorder=1, label="Volume")

    ax1.plot(df["time"], df["value"], color=color_line, lw=lw, solid_capstyle="round", label="Index", zorder=3)
    ax1.set_title(f"{index_key.upper()} ({label})", color="#ffb6c1")
    ax1.set_xlabel("Time" if mode == "1d" else "Date")
    ax1.set_ylabel("Index Value")

    format_time_axis(ax1, mode, tz)
    apply_y_padding(ax1, df["value"])
    if frame is not None:
        ax1.set_xlim(frame)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outpath = f"{OUTPUT_DIR}/{index_key}_{label}.png"
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()
    log(f"saved: {outpath}")

# ============================================================
#  メイン
# ============================================================

def main():
    key = INDEX_KEY  # 例: rbank9
    MP = MARKET
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    intraday_path = find_intraday(OUTPUT_DIR, key)
    history_path = find_history(OUTPUT_DIR, key)

    intraday = read_any(intraday_path, MP["RAW_TZ_INTRADAY"], MP["DISPLAY_TZ"]) if intraday_path else pd.DataFrame()
    history = read_any(history_path, MP["RAW_TZ_HISTORY"], MP["DISPLAY_TZ"]) if history_path else pd.DataFrame()

    daily_all = to_daily(history if not history.empty else intraday, MP["DISPLAY_TZ"])

    # ---- 1d（当日セッション切り出し）
    if not intraday.empty:
        last_ts = intraday["time"].max()  # すでに DISPLAY_TZ 変換済み
        start_jst, end_jst = jst_session_frame(last_ts, MP["SESSION_START"], MP["SESSION_END"], MP["SESSION_TZ"])
        # intraday["time"] は DISPLAY_TZ（=JST）なのでそのまま比較
        mask = (intraday["time"] >= start_jst) & (intraday["time"] <= end_jst)
        df_1d = intraday.loc[mask].copy()
        frame_1d = (start_jst, end_jst)
    else:
        df_1d = pd.DataFrame()
        frame_1d = None

    plot_df(df_1d, key, "1d", "1d", MP["DISPLAY_TZ"], frame=frame_1d)

    # ---- 7d / 1m / 1y
    now = pd.Timestamp.now(tz=MP["DISPLAY_TZ"])
    for label, days in [("7d", 7), ("1m", 31), ("1y", 365)]:
        sub = daily_all[daily_all["time"] >= (now - timedelta(days=days))] if not daily_all.empty else pd.DataFrame()
        plot_df(sub, key, label, "long", MP["DISPLAY_TZ"])

if __name__ == "__main__":
    main()
