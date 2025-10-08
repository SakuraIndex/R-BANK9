#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate long-term charts for R-BANK9 (1d / 7d / 1m / 1y).

- 1d: 当日 9:00–15:30(JST) の等加重指数（板形式CSVにも対応）
- 7d/1m/1y: docs/outputs/<key>_history.csv（date,value）を使用
- 生成先: docs/outputs/<key>_{1d|7d|1m|1y}.png
- 最終実行時刻: docs/outputs/_last_run.txt
"""

from __future__ import annotations

import os
import io
import math
import datetime as dt
from typing import Optional, Tuple, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========= 設定 =========

INDEX_KEY = os.getenv("INDEX_KEY", "rbank9").lower()  # 例: rbank9
OUTPUT_DIR = os.path.join("docs", "outputs")

# 表示タイムゾーン（日本株）
DISPLAY_TZ = "Asia/Tokyo"

# 当日セッション（JST）
SESSION_START: Tuple[int, int] = (9, 0)    # 09:00
SESSION_END:   Tuple[int, int] = (15, 30)  # 15:30

# 入力ファイル（存在しない場合は空として扱う）
INTRADAY_CSV = os.path.join(OUTPUT_DIR, f"{INDEX_KEY}_intraday.csv")
HISTORY_CSV  = os.path.join(OUTPUT_DIR, f"{INDEX_KEY}_history.csv")

# タイトル色（ダーク背景前提）
TITLE_COLOR = "#F5B6C7"


# ========= 汎用ユーティリティ =========

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def ensure_tz(series: pd.Series, tz: str) -> pd.Series:
    """Series[datetime64] を tz-aware にして tz へ変換"""
    if not isinstance(series, pd.Series):
        return series
    # pandas 2.x 互換: tzinfo は属性としては持たないことが多いので dtype 判定
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        if series.dt.tz is None:
            # 素の naive → とりあえず tz ローカライズ（ここでは tz を直接付与）
            series = series.dt.tz_localize(tz)
        else:
            series = series.dt.tz_convert(tz)
    return series


def parse_time_any(x, raw_tz: str, display_tz: str) -> pd.Timestamp | pd.NaT:
    """文字列/数値/epoch を pd.Timestamp(表示TZ) に変換"""
    try:
        ts = pd.to_datetime(x, utc=False, errors="coerce")
    except Exception:
        return pd.NaT
    if pd.isna(ts):
        return pd.NaT
    # naive なら raw_tz を付与
    if getattr(ts, "tzinfo", None) is None and not getattr(ts, "tz", None):
        try:
            ts = ts.tz_localize(raw_tz)
        except Exception:
            # 万一失敗したら UTC → convert
            ts = ts.tz_localize("UTC")
    return ts.tz_convert(display_tz)


def pick_time_col(cols_lower: List[str]) -> Optional[str]:
    """タイムスタンプらしき列名を推定"""
    candidates = ("time", "timestamp", "date", "datetime")
    for name in candidates:
        if name in cols_lower:
            return name
    # Unnamed: 0（インデックス書き出し）を time とみなす
    for c in cols_lower:
        if c.startswith("unnamed"):
            return c
    # あいまい一致
    fuzzy = [c for c in cols_lower if ("time" in c) or ("date" in c)]
    return fuzzy[0] if fuzzy else None


def pick_value_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        lc = str(c).lower()
        if any(k in lc for k in ("value", "index", "score")):
            return c
    return None


def pick_volume_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if "volume" in str(c).lower():
            return c
    return None


# ========= データ読み込み =========

def read_intraday(path: str, raw_tz: str = DISPLAY_TZ) -> pd.DataFrame:
    """
    intraday CSV を読み、["time","value","volume"] に正規化して返す。
    - time/value/volume 形式 or 板形式（複数銘柄列）を両対応。
    - 板形式は数値化できる列の等加重平均を value にする。
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=["time", "value", "volume"])

    df = pd.read_csv(path)

    # 列名を小文字化
    raw_cols = list(df.columns)
    cols_lower = [str(c).strip().lower() for c in raw_cols]
    df.columns = cols_lower

    # 時刻列を推定
    tcol = pick_time_col(cols_lower)
    if tcol is None or tcol not in df.columns:
        raise KeyError(f"No time-like column found. columns={list(df.columns)}")

    # 「単一 value/volume 列」形式の判定
    vcol = pick_value_col(df)
    volcol = pick_volume_col(df)

    out = pd.DataFrame()
    out["time"] = df[tcol].apply(lambda x: parse_time_any(x, raw_tz, DISPLAY_TZ))
    out["time"] = ensure_tz(out["time"], DISPLAY_TZ)

    if (vcol is not None) and (vcol in df.columns):
        out["value"] = pd.to_numeric(df[vcol], errors="coerce")
    else:
        # 板形式：time 以外で数値化できる列を平均
        num_cols: List[str] = []
        for c in df.columns:
            if c == tcol:
                continue
            as_num = pd.to_numeric(df[c], errors="coerce")
            if as_num.notna().sum() > 0:
                num_cols.append(c)
        if not num_cols:
            return pd.DataFrame(columns=["time", "value", "volume"])
        vals_df = df[num_cols].apply(lambda s: pd.to_numeric(s, errors="coerce"))
        out["value"] = vals_df.mean(axis=1)

    if (volcol is not None) and (volcol in df.columns):
        out["volume"] = pd.to_numeric(df[volcol], errors="coerce").fillna(0)
    else:
        out["volume"] = 0

    out = (
        out.dropna(subset=["time", "value"])
           .sort_values("time")
           .reset_index(drop=True)
    )
    return out


def read_history(path: str) -> pd.DataFrame:
    """
    docs/outputs/<key>_history.csv (date,value) を読み、
    当日 15:30:00 JST のタイムスタンプに変換して返す。

    !!! 修正点 !!!
    Timedelta を "hh:mm:ss" 形式で作成（以前の "hh:mm" で落ちていた問題を解消）
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=["time", "value", "volume"])

    df = pd.read_csv(path)
    if not {"date", "value"}.issubset(set(df.columns)):
        return pd.DataFrame(columns=["time", "value", "volume"])

    # 営業日の日付を JST の 00:00 に
    t = pd.to_datetime(df["date"], errors="coerce")
    t = t.dt.tz_localize(DISPLAY_TZ)

    # 15:30:00 を加算（hh:mm:ss で作る）
    sess_end = pd.to_timedelta(f"{SESSION_END[0]:02d}:{SESSION_END[1]:02d}:00")
    t = t + sess_end

    out = pd.DataFrame({
        "time": t,
        "value": pd.to_numeric(df["value"], errors="coerce"),
        "volume": 0,
    }).dropna(subset=["time", "value"]).sort_values("time").reset_index(drop=True)

    return out


# ========= 可視化 =========

def _mpl_dark():
    plt.rcParams.update({
        "figure.facecolor": "#0E1117",
        "axes.facecolor": "#0E1117",
        "savefig.facecolor": "#0E1117",
        "axes.edgecolor": "#2D3440",
        "axes.labelcolor": "#C9D1D9",
        "xtick.color": "#C9D1D9",
        "ytick.color": "#C9D1D9",
        "grid.color": "#2D3440",
        "text.color": "#C9D1D9",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def _save(fig, path: str):
    _ensure_dir(os.path.dirname(path))
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_1d(intraday: pd.DataFrame, key: str):
    # 当日 9:00–15:30 に絞り込み
    if intraday.empty:
        # 空でも軸だけ出す
        fig, ax = plt.subplots(figsize=(12, 6))
        _mpl_dark()
        ax.set_title(f"{key.upper()} (1d)", color=TITLE_COLOR, fontsize=18)
        ax.set_xlabel("Time")
        ax.set_ylabel("Index Value")
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", alpha=0.5, fontsize=18)
        _save(fig, os.path.join(OUTPUT_DIR, f"{key}_1d.png"))
        return

    ts = intraday["time"].dt.tz_convert(DISPLAY_TZ)
    day = ts.dt.date.iloc[-1]  # 末尾の営業日
    day0 = pd.Timestamp(dt.date(day.year, day.month, day.day), tz=DISPLAY_TZ)

    t_start = day0 + pd.to_timedelta(f"{SESSION_START[0]:02d}:{SESSION_START[1]:02d}:00")
    t_end   = day0 + pd.to_timedelta(f"{SESSION_END[0]:02d}:{SESSION_END[1]:02d}:00")

    mask = (ts >= t_start) & (ts <= t_end)
    df = intraday.loc[mask].copy()
    if df.empty:
        df = intraday.copy()

    fig, ax = plt.subplots(figsize=(12, 6))
    _mpl_dark()

    ax.plot(df["time"], df["value"], linewidth=2.4, color="#2FD8C7")
    ax.set_title(f"{key.upper()} (1d)", color=TITLE_COLOR, fontsize=18)
    ax.set_xlabel("Time")
    ax.set_ylabel("Index Value")
    fig.autofmt_xdate()

    _save(fig, os.path.join(OUTPUT_DIR, f"{key}_1d.png"))


def plot_window(history: pd.DataFrame, key: str, days: int, suffix: str):
    """
    history から直近 days 日を表示（営業日ベースの単純切り出し）
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    _mpl_dark()

    ax.set_title(f"{key.upper()} ({suffix})", color=TITLE_COLOR, fontsize=18)
    ax.set_xlabel("Date" if suffix != "1d" else "Time")
    ax.set_ylabel("Index Value")

    if history.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", alpha=0.5, fontsize=18)
        _save(fig, os.path.join(OUTPUT_DIR, f"{key}_{suffix}.png"))
        return

    # 末尾基準で days 日分
    end_ts = history["time"].max()
    start_ts = end_ts - pd.Timedelta(days=days)
    df = history.loc[history["time"] >= start_ts].copy()

    if df.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", alpha=0.5, fontsize=18)
        _save(fig, os.path.join(OUTPUT_DIR, f"{key}_{suffix}.png"))
        return

    ax.plot(df["time"], df["value"], linewidth=2.2, color="#2FD8C7")
    fig.autofmt_xdate()
    _save(fig, os.path.join(OUTPUT_DIR, f"{key}_{suffix}.png"))


# ========= メイン処理 =========

def main():
    _ensure_dir(OUTPUT_DIR)

    # 1) intraday → 1d
    try:
        intraday = read_intraday(INTRADAY_CSV, raw_tz=DISPLAY_TZ)
    except Exception as e:
        print(f"WARN: read_intraday failed: {e}")
        intraday = pd.DataFrame(columns=["time", "value", "volume"])
    plot_1d(intraday, INDEX_KEY)

    # 2) history → 7d / 1m(30d) / 1y(365d)
    try:
        history = read_history(HISTORY_CSV)
    except Exception as e:
        print(f"WARN: read_history failed: {e}")
        history = pd.DataFrame(columns=["time", "value", "volume"])

    plot_window(history, INDEX_KEY, days=7,   suffix="7d")
    plot_window(history, INDEX_KEY, days=30,  suffix="1m")
    plot_window(history, INDEX_KEY, days=365, suffix="1y")

    # 3) 最終実行マーカー
    with open(os.path.join(OUTPUT_DIR, "_last_run.txt"), "w", encoding="utf-8") as f:
        now = pd.Timestamp.now(tz=DISPLAY_TZ)
        f.write(str(now))


if __name__ == "__main__":
    main()
