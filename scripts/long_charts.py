#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate long-term charts (1d/7d/1m/1y) for Sakura Index.
- 1d は 始値<終値: 青緑 / 始値>終値: 赤 / 同値: グレー
- 出来高があれば薄い棒で重ね描き
- 板形式CSV（銘柄ごとに列）にも対応：数値化できる列の等加重平均で value を作成
- 先頭列が 'Unnamed: 0'（インデックス書き出し）でも時刻列として自動認識
- ‘#’ で始まる列は自動除外
- R-BANK9 は日本株：JST 9:00–15:30 の当日セッションで 1d を切り出し
出力: docs/outputs/<index_key>_{1d|7d|1m|1y}.png
"""

from __future__ import annotations

import os
import re
from typing import Optional, List, Tuple
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =========================
#  基本設定
# =========================

OUTPUT_DIR = "docs/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 画像の色など
COLOR_PRICE_DEFAULT = "#ff99cc"  # 長期線
COLOR_VOLUME = "#7f8ca6"
COLOR_UP = "#00C2A0"   # 陽線（上昇）
COLOR_DOWN = "#FF4C4C" # 陰線（下落）
COLOR_EQUAL = "#CCCCCC"

# 画面テーマ
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

def log(msg: str) -> None:
    print(f"[long_charts] {msg}")

# ---- スタンプ（強制差分 & どこに保存したかを可視化）
STAMP_PATH = os.path.join(OUTPUT_DIR, "_last_run.txt")
def _touch_stamp() -> None:
    ts = pd.Timestamp.now(tz="Asia/Tokyo")
    with open(STAMP_PATH, "w", encoding="utf-8") as f:
        f.write(str(ts))
    print(f"[INFO] wrote stamp: {os.path.abspath(STAMP_PATH)}")

# =========================
#  マーケットプロファイル
# =========================

def market_profile(index_key: str) -> dict:
    k = (index_key or "").lower()
    # R-BANK9：日本株（JST 9:00-15:30）
    if k in ("rbank9", "r-bank9", "rbk9"):
        return dict(
            RAW_TZ_INTRADAY="Asia/Tokyo",
            RAW_TZ_HISTORY="Asia/Tokyo",
            DISPLAY_TZ="Asia/Tokyo",
            SESSION_TZ="Asia/Tokyo",
            SESSION_START=(9, 0),
            SESSION_END=(15, 30),
        )
    # 既定（JST現物）
    return dict(
        RAW_TZ_INTRADAY="Asia/Tokyo",
        RAW_TZ_HISTORY="Asia/Tokyo",
        DISPLAY_TZ="Asia/Tokyo",
        SESSION_TZ="Asia/Tokyo",
        SESSION_START=(9, 0),
        SESSION_END=(15, 0),
    )

# =========================
#  入出力ヘルパ
# =========================

def _first(paths: List[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
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

# =========================
#  CSV 解析
# =========================

TIME_CANDIDATES = ["datetime", "time", "timestamp", "date", "unnamed: 0"]

def ensure_tz(series: pd.Series, tz: str) -> pd.Series:
    if not isinstance(series, pd.Series):
        return series
    try:
        # pandas は tz-aware なら tz_convert、naive なら tz_localize が必要
        if getattr(series.dt, "tz", None) is None:
            return series.dt.tz_localize(tz)
        return series.dt.tz_convert(tz)
    except Exception:
        return series

def parse_time_any(x, raw_tz: str, display_tz: str) -> pd.Timestamp | pd.NaT:
    if pd.isna(x):
        return pd.NaT
    s = str(x).strip()

    # UNIX 秒（10桁）
    if re.fullmatch(r"\d{10}", s):
        try:
            return pd.Timestamp(int(s), unit="s", tz="UTC").tz_convert(display_tz)
        except Exception:
            return pd.NaT

    # 一般的な parse
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
        if name in cols_lower:
            return name
    # あいまい検索
    fuzzy = [c for c in cols_lower if ("time" in c) or ("date" in c)]
    return fuzzy[0] if fuzzy else None

def pick_value_col(df: pd.DataFrame) -> Optional[str]:
    lc = [c.lower() for c in df.columns]
    for k in ["value", "close", "index", "score", "終値"]:
        if k in lc:
            return df.columns[lc.index(k)]
    # 数値列の先頭
    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return nums[0] if nums else None

def pick_volume_col(df: pd.DataFrame) -> Optional[str]:
    lc = [c.lower() for c in df.columns]
    for k in ["volume", "vol", "出来高"]:
        if k in lc:
            return df.columns[lc.index(k)]
    return None

def read_any(path: Optional[str], raw_tz: str, display_tz: str) -> pd.DataFrame:
    """
    列名を正規化してから、(1) time/value[/volume] 形式  または
    (2) 銘柄ごとの板状形式（数値列の等加重平均）を受け付ける。
    返り値は ["time","value","volume"] を持つ DataFrame。
    """
    if not path:
        return pd.DataFrame(columns=["time", "value", "volume"])

    df = pd.read_csv(path)

    # 列名 正規化（小文字・trim）
    raw_cols = list(df.columns)
    cols_lower = [str(c).strip().lower() for c in raw_cols]
    df.columns = cols_lower

    # ‘#’ で始まる列は除外（コメント列）
    drop_cols = [c for c in df.columns if c.startswith("#")]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    # 時刻列
    tcol = pick_time_col(list(df.columns))
    if tcol is None:
        raise KeyError(f"No time-like column found. columns={list(df.columns)}")

    # 既に value/volume が単一列で存在する形式
    vcol = pick_value_col(df)
    volcol = pick_volume_col(df)
    if (vcol is not None) and (vcol in df.columns) and ((volcol is None) or (volcol in df.columns)):
        out = pd.DataFrame()
        out["time"] = df[tcol].apply(lambda x: parse_time_any(x, raw_tz, display_tz))
        out["value"] = pd.to_numeric(df[vcol], errors="coerce")
        out["volume"] = pd.to_numeric(df[volcol], errors="coerce") if (volcol and volcol in df.columns) else 0
        out = out.dropna(subset=["time", "value"]).sort_values("time").reset_index(drop=True)
        return out

    # 盤面形式：時刻列以外で数値化できる列を平均
    num_cols: List[str] = []
    for c in df.columns:
        if c == tcol:
            continue
        as_num = pd.to_numeric(df[c], errors="coerce")
        if as_num.notna().sum() > 0:
            num_cols.append(c)

    if len(num_cols) == 0:
        return pd.DataFrame(columns=["time", "value", "volume"])

    out = pd.DataFrame()
    out["time"] = df[tcol].apply(lambda x: parse_time_any(x, raw_tz, display_tz))

    # 列ごとに数値化してから平均（coerce→mean）
    vals_df = df[num_cols].apply(lambda s: pd.to_numeric(s, errors="coerce"))
    out["value"] = vals_df.mean(axis=1)

    out["volume"] = 0
    out = out.dropna(subset=["time", "value"]).sort_values("time").reset_index(drop=True)
    return out

def to_daily(df: pd.DataFrame, display_tz: str) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.copy()
    # 念のため tz を保証
    if getattr(d["time"].dt, "tz", None) is None:
        d["time"] = d["time"].dt.tz_localize(display_tz)
    d["date"] = d["time"].dt.tz_convert(display_tz).dt.date
    g = d.groupby("date", as_index=False).agg({"value": "last", "volume": "sum"})
    g["time"] = pd.to_datetime(g["date"]).dt.tz_localize(display_tz)
    return g[["time", "value", "volume"]]

# =========================
#  描画補助
# =========================

def format_time_axis(ax, mode: str, tz: str) -> None:
    if mode == "1d":
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1, tz=tz))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tz))
    else:
        loc = mdates.AutoDateLocator(minticks=3, maxticks=6, tz=tz)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))

def apply_y_padding(ax, series: pd.Series) -> None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        ax.set_ylim(0, 1)
        return
    lo, hi = s.min(), s.max()
    pad = (hi - lo) * 0.08 if hi != lo else max(abs(lo) * 0.02, 0.5)
    ax.set_ylim(lo - pad, hi + pad)

def session_frame(base_ts_jst: pd.Timestamp,
                  session_tz: str, display_tz: str,
                  start_hm: Tuple[int, int], end_hm: Tuple[int, int]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """当日セッションの開始/終了の表示TZ境界（base_ts の日を基準）"""
    stz = session_tz
    et = base_ts_jst.tz_convert(stz)
    d = et.date()
    start_et = pd.Timestamp(d.year, d.month, d.day, start_hm[0], start_hm[1], tz=stz)
    end_et   = pd.Timestamp(d.year, d.month, d.day, end_hm[0], end_hm[1], tz=stz)
    return start_et.tz_convert(display_tz), end_et.tz_convert(display_tz)

def plot_df(df: pd.DataFrame, index_key: str, label: str, mode: str, tz: str,
            frame: Tuple[pd.Timestamp, pd.Timestamp] | None = None) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outpath = f"{OUTPUT_DIR}/{index_key}_{label}.png"

    fig, ax1 = plt.subplots(figsize=(9.5, 4.8))
    ax1.grid(True, alpha=0.3)

    if df.empty:
        ax1.text(0.5, 0.5, "No data", transform=ax1.transAxes, ha="center", va="center",
                 color="#8a93b2", fontsize=22, alpha=0.8)
        ax1.set_title(f"{index_key.upper()} ({label})", color="#ffb6c1")
        ax1.set_xlabel("Time" if mode == "1d" else "Date")
        ax1.set_ylabel("Index Value")
        format_time_axis(ax1, mode, tz)
        plt.tight_layout()
        plt.savefig(outpath, dpi=180)
        plt.close()
        print(f"[SAVE] {os.path.abspath(outpath)} (empty)")
        _touch_stamp()
        return

    # 1d の線色は始値/終値で変化
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
    if "volume" in df.columns and np.nan_to_num(df["volume"].values).sum() > 0:
        ax2 = ax1.twinx()
        ax2.bar(df["time"], df["volume"],
                width=0.9 if mode == "1d" else 0.8,
                color=COLOR_VOLUME, alpha=0.35, zorder=1, label="Volume")

    ax1.plot(df["time"], df["value"], color=color_line, lw=lw,
             solid_capstyle="round", label="Index", zorder=3)
    ax1.set_title(f"{index_key.upper()} ({label})", color="#ffb6c1")
    ax1.set_xlabel("Time" if mode == "1d" else "Date")
    ax1.set_ylabel("Index Value")

    format_time_axis(ax1, mode, tz)
    apply_y_padding(ax1, df["value"])
    if frame is not None:
        ax1.set_xlim(frame)

    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()
    print(f"[SAVE] {os.path.abspath(outpath)}")
    _touch_stamp()

# =========================
#  メイン
# =========================

def main() -> None:
    index_key = os.environ.get("INDEX_KEY", "rbank9").lower()
    MP = market_profile(index_key)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 入力ファイルの検出（docs/outputs 以下）
    intraday_path = find_intraday(OUTPUT_DIR, index_key)
    history_path  = find_history(OUTPUT_DIR, index_key)
    print(f"[INFO] intraday_path={intraday_path}")
    print(f"[INFO] history_path={history_path}")

    intraday = read_any(intraday_path, MP["RAW_TZ_INTRADAY"], MP["DISPLAY_TZ"]) if intraday_path else pd.DataFrame()
    history  = read_any(history_path,  MP["RAW_TZ_HISTORY"],  MP["DISPLAY_TZ"]) if history_path  else pd.DataFrame()
    daily_all = to_daily(history if not history.empty else intraday, MP["DISPLAY_TZ"])

    # ---- 1d：当日セッションで切り出し（JST 9:00–15:30）
    if not intraday.empty:
        last_ts = intraday["time"].max()
        start_jst, end_jst = session_frame(
            last_ts, MP["SESSION_TZ"], MP["DISPLAY_TZ"],
            MP["SESSION_START"], MP["SESSION_END"]
        )
        mask = (intraday["time"] >= start_jst) & (intraday["time"] <= end_jst)
        df_1d = intraday.loc[mask].copy()
        frame_1d = (start_jst, end_jst)
        # 念のため、1日の値が「平均化板データ」で極端に小さくなった場合もそのまま描画
    else:
        df_1d = pd.DataFrame(columns=["time", "value", "volume"])
        frame_1d = None

    plot_df(df_1d, index_key, "1d", "1d", MP["DISPLAY_TZ"], frame=frame_1d)

    # ---- 7d / 1m / 1y（終値ベースの長期）
    now = pd.Timestamp.now(tz=MP["DISPLAY_TZ"])
    for label, days in [("7d", 7), ("1m", 31), ("1y", 365)]:
        sub = daily_all[daily_all["time"] >= (now - timedelta(days=days))] if not daily_all.empty else pd.DataFrame()
        plot_df(sub, index_key, label, "long", MP["DISPLAY_TZ"])

if __name__ == "__main__":
    main()
