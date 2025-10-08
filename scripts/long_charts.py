#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate long-term charts (1d/7d/1m/1y) for R-BANK9.
- 1d: 当日セッション(JST 9:00-15:30)の intraday を描画
- 7d/1m/1y: docs/outputs/<key>_history.csv(date,value) を読み、JST にローカライズして描画
- 板形式 intraday にも対応（等加重平均）

出力: docs/outputs/<index_key>_{1d|7d|1m|1y}.png
"""

from __future__ import annotations
import os
import re
from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 基本
INDEX_KEY = (os.environ.get("INDEX_KEY") or "rbank9").lower()
OUT_DIR = "docs/outputs"
DISPLAY_TZ = "Asia/Tokyo"
SESSION_TZ = "Asia/Tokyo"
SESSION_START = (9, 0)
SESSION_END   = (15, 30)

# 色など
COLOR_PRICE_DEFAULT = "#66e0c2"
COLOR_VOLUME = "#7f8ca6"
COLOR_UP = "#00C2A0"
COLOR_DOWN = "#FF4C4C"
COLOR_EQUAL = "#CCCCCC"

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

# ---------- 入出力補助

def parse_time_any(x, raw_tz: str, display_tz: str):
    if pd.isna(x):
        return pd.NaT
    s = str(x).strip()
    if re.fullmatch(r"\d{10}", s):
        return pd.Timestamp(int(s), unit="s", tz="UTC").tz_convert(display_tz)
    if re.fullmatch(r"\d{13}", s):
        return pd.Timestamp(int(s), unit="ms", tz="UTC").tz_convert(display_tz)
    t = pd.to_datetime(s, errors="coerce")
    if pd.isna(t):
        return pd.NaT
    if t.tzinfo is None:
        t = t.tz_localize(raw_tz)
    return t.tz_convert(display_tz)

def pick_time_col(cols_lower):
    for k in ["time", "timestamp", "datetime", "date", "unnamed: 0"]:
        if k in cols_lower:
            return k
    fuzzy = [c for c in cols_lower if ("time" in c) or ("date" in c)]
    return fuzzy[0] if fuzzy else None

def read_intraday(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["time","value","volume"])
    df = pd.read_csv(path)
    df = df[[c for c in df.columns if not str(c).strip().startswith("#")]]
    raw_cols = list(df.columns)
    cols_lower = [str(c).strip().lower() for c in raw_cols]
    df.columns = cols_lower

    tcol = pick_time_col(cols_lower)
    if tcol is None:
        raise KeyError(f"No time-like column found. columns={list(df.columns)}")

    # 値列推定
    vcol = None
    for k in ["value", "price", "index", "score", "終値"]:
        if k in cols_lower:
            vcol = k
            break

    out = pd.DataFrame()
    out["time"] = df[tcol].apply(lambda x: parse_time_any(x, DISPLAY_TZ, DISPLAY_TZ))

    if vcol is not None:
        out["value"] = pd.to_numeric(df[vcol], errors="coerce")
    else:
        num_cols = []
        for c in df.columns:
            if c == tcol:
                continue
            as_num = pd.to_numeric(df[c], errors="coerce")
            if as_num.notna().sum() > 0:
                num_cols.append(c)
        if not num_cols:
            return pd.DataFrame(columns=["time","value","volume"])
        vals_df = df[num_cols].apply(lambda s: pd.to_numeric(s, errors="coerce"))
        out["value"] = vals_df.mean(axis=1)

    out["volume"] = 0
    out = out.dropna(subset=["time","value"]).sort_values("time").reset_index(drop=True)
    return out

def read_history(path: str) -> pd.DataFrame:
    """
    docs/outputs/<key>_history.csv (date,value) を読み、time(JST) 列に変換
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=["time","value","volume"])
    df = pd.read_csv(path)
    if not {"date","value"}.issubset(set(df.columns)):
        # 古い形式は無視
        return pd.DataFrame(columns=["time","value","volume"])
    # date を JST の 15:30 にしておく（日中での表示ズレ回避のため）
    t = pd.to_datetime(df["date"], errors="coerce")
    t = t.dt.tz_localize(DISPLAY_TZ).dt.tz_convert(DISPLAY_TZ)
    t = t + pd.to_timedelta(f"{SESSION_END[0]}:{SESSION_END[1]}")
    out = pd.DataFrame({
        "time": t,
        "value": pd.to_numeric(df["value"], errors="coerce"),
        "volume": 0,
    }).dropna(subset=["time","value"]).sort_values("time").reset_index(drop=True)
    return out

def session_frame_for(ts_jst: pd.Timestamp):
    j = ts_jst.tz_convert(SESSION_TZ)
    d = j.date()
    start = pd.Timestamp(d.year, d.month, d.day, SESSION_START[0], SESSION_START[1], tz=SESSION_TZ).tz_convert(DISPLAY_TZ)
    end   = pd.Timestamp(d.year, d.month, d.day, SESSION_END[0],   SESSION_END[1],   tz=SESSION_TZ).tz_convert(DISPLAY_TZ)
    return start, end

# ---------- 描画

def format_time_axis(ax, mode):
    if mode == "1d":
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1, tz=DISPLAY_TZ))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=DISPLAY_TZ))
    else:
        loc = mdates.AutoDateLocator(minticks=3, maxticks=6, tz=DISPLAY_TZ)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))

def apply_y_padding(ax, s: pd.Series):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        ax.set_ylim(0, 1)
        return
    lo, hi = s.min(), s.max()
    pad = (hi - lo) * 0.08 if hi != lo else max(abs(lo) * 0.02, 0.5)
    ax.set_ylim(lo - pad, hi + pad)

def plot_df(df, label, mode, frame=None):
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.grid(True, alpha=0.3)

    if df.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", fontsize=20, color="#9aa4bf")
        ax.set_title(f"{INDEX_KEY.upper()} ({label})", color="#ffb6c1")
        ax.set_xlabel("Time" if mode == "1d" else "Date")
        ax.set_ylabel("Index Value")
        format_time_axis(ax, mode)
        os.makedirs(OUT_DIR, exist_ok=True)
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/{INDEX_KEY}_{label}.png", dpi=180)
        plt.close()
        log(f"saved empty: {OUT_DIR}/{INDEX_KEY}_{label}.png")
        return

    if mode == "1d":
        open_p = df["value"].iloc[0]
        close_p = df["value"].iloc[-1]
        color = COLOR_UP if close_p > open_p else (COLOR_DOWN if close_p < open_p else COLOR_EQUAL)
        lw = 2.2
    else:
        color = COLOR_PRICE_DEFAULT
        lw = 1.8

    ax.plot(df["time"], df["value"], color=color, lw=lw, solid_capstyle="round")
    ax.set_title(f"{INDEX_KEY.upper()} ({label})", color="#ffb6c1")
    ax.set_xlabel("Time" if mode == "1d" else "Date")
    ax.set_ylabel("Index Value")
    format_time_axis(ax, mode)
    apply_y_padding(ax, df["value"])
    if frame is not None:
        ax.set_xlim(frame)

    os.makedirs(OUT_DIR, exist_ok=True)
    plt.tight_layout()
    outpath = f"{OUT_DIR}/{INDEX_KEY}_{label}.png"
    plt.savefig(outpath, dpi=180)
    plt.close()
    log(f"saved: {outpath}")

# ---------- メイン

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    intraday_path = f"{OUT_DIR}/{INDEX_KEY}_intraday.csv"
    history_path  = f"{OUT_DIR}/{INDEX_KEY}_history.csv"

    intraday = read_intraday(intraday_path)
    history  = read_history(history_path)

    # 1d
    if not intraday.empty:
        last_ts = intraday["time"].max()
        start, end = session_frame_for(last_ts)
        day = intraday[(intraday["time"] >= start) & (intraday["time"] <= end)].copy()
        frame = (start, end)
    else:
        day = pd.DataFrame(columns=["time","value","volume"])
        frame = None
    plot_df(day, "1d", "1d", frame)

    # 7d/1m/1y (history ベース)
    now = pd.Timestamp.now(tz=DISPLAY_TZ)
    for label, days in [("7d", 7), ("1m", 31), ("1y", 365)]:
        sub = history[history["time"] >= (now - pd.Timedelta(days=days))].copy()
        plot_df(sub, label, "long")

    # 最終実行マーカー
    with open(f"{OUT_DIR}/_last_run.txt", "w") as f:
        f.write(str(pd.Timestamp.now(tz=DISPLAY_TZ)))

if __name__ == "__main__":
    main()
