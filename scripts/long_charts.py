#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate long-term charts (1d/7d/1m/1y) for Sakura Index series.
Automatically adapts session & timezone based on INDEX_KEY.
Includes 1d color: green(up) / red(down) / gray(equal).
"""

import os
import re
import glob
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ============================================================
#  基本設定
# ============================================================

OUTPUT_DIR = "docs/outputs"

# 色設定
COLOR_PRICE_DEFAULT = "#ff99cc"
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

# ============================================================
#  市場セッション定義
# ============================================================

def market_profile(index_key: str):
    """
    各指数ごとの取引セッション設定
    """
    k = (index_key or "").lower()

    # ✅ AIN-10：米国株系（ET 9:30〜16:00 → JST表示）
    if k == "ain10":
        return dict(
            RAW_TZ_INTRADAY="America/New_York",
            RAW_TZ_HISTORY="Asia/Tokyo",
            DISPLAY_TZ="Asia/Tokyo",
            SESSION_TZ="America/New_York",
            SESSION_START=(9, 30),
            SESSION_END=(16, 0),
        )

    # ✅ Astra4：米国株中心テーマ（ET 9:30〜16:00 → JST表示）
    if k == "astra4":
        return dict(
            RAW_TZ_INTRADAY="America/New_York",
            RAW_TZ_HISTORY="Asia/Tokyo",
            DISPLAY_TZ="Asia/Tokyo",
            SESSION_TZ="America/New_York",
            SESSION_START=(9, 30),
            SESSION_END=(16, 0),
        )

    # ✅ S-COIN+：日本株（暗号資産関連上場企業）9:00〜15:30 JST
    if k in ("scoin+", "scoin_plus", "scoinplus", "s-coin+"):
        return dict(
            RAW_TZ_INTRADAY="Asia/Tokyo",
            RAW_TZ_HISTORY="Asia/Tokyo",
            DISPLAY_TZ="Asia/Tokyo",
            SESSION_TZ="Asia/Tokyo",
            SESSION_START=(9, 0),
            SESSION_END=(15, 30),
        )

    # ✅ R-BANK9：地方銀行指数（日本株）9:00〜15:00 JST
    if k == "rbank9":
        return dict(
            RAW_TZ_INTRADAY="Asia/Tokyo",
            RAW_TZ_HISTORY="Asia/Tokyo",
            DISPLAY_TZ="Asia/Tokyo",
            SESSION_TZ="Asia/Tokyo",
            SESSION_START=(9, 0),
            SESSION_END=(15, 0),
        )

    # fallback
    return dict(
        RAW_TZ_INTRADAY="Asia/Tokyo",
        RAW_TZ_HISTORY="Asia/Tokyo",
        DISPLAY_TZ="Asia/Tokyo",
        SESSION_TZ="Asia/Tokyo",
        SESSION_START=(9, 0),
        SESSION_END=(15, 0),
    )

# ============================================================
#  入出力関数群
# ============================================================

def _first(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def find_intraday(base, key):
    return _first([
        f"{base}/{key}_intraday.csv",
        f"{base}/{key}_intraday.txt",
    ])

def find_history(base, key):
    return _first([
        f"{base}/{key}_history.csv",
        f"{base}/{key}_history.txt",
    ])

def parse_time_any(x, raw_tz, display_tz):
    if pd.isna(x):
        return pd.NaT
    s = str(x)
    if re.fullmatch(r"\d{10}", s):
        return pd.Timestamp(int(s), unit="s", tz="UTC").tz_convert(display_tz)
    try:
        t = pd.to_datetime(s)
        if t.tzinfo is None:
            t = t.tz_localize(raw_tz)
        return t.tz_convert(display_tz)
    except Exception:
        return pd.NaT

def pick_value_col(df):
    df.columns = [c.lower() for c in df.columns]
    for k in ["close", "price", "value", "index", "終値"]:
        if k in df.columns:
            return k
    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return nums[0] if nums else df.columns[0]

def pick_volume_col(df):
    df.columns = [c.lower() for c in df.columns]
    for k in ["volume", "vol", "出来高"]:
        if k in df.columns:
            return k
    return None

def read_any(path, raw_tz, display_tz):
    if not path:
        return pd.DataFrame(columns=["time","value","volume"])
    df = pd.read_csv(path)
    tcol = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()][0]
    vcol = pick_value_col(df)
    volcol = pick_volume_col(df)
    out = pd.DataFrame()
    out["time"] = df[tcol].apply(lambda x: parse_time_any(x, raw_tz, display_tz))
    out["value"] = pd.to_numeric(df[vcol], errors="coerce")
    out["volume"] = pd.to_numeric(df[volcol], errors="coerce") if volcol else 0
    return out.dropna(subset=["time","value"]).sort_values("time").reset_index(drop=True)

def to_daily(df, display_tz):
    if df.empty:
        return df
    d = df.copy()
    d["date"] = d["time"].dt.tz_convert(display_tz).dt.date
    g = d.groupby("date", as_index=False).agg({"value":"last","volume":"sum"})
    g["time"] = pd.to_datetime(g["date"]).dt.tz_localize(display_tz)
    return g[["time","value","volume"]]

# ============================================================
#  グラフ描画補助
# ============================================================

def format_time_axis(ax, mode, tz):
    if mode == "1d":
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1, tz=tz))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tz))
    else:
        loc = mdates.AutoDateLocator(minticks=3, maxticks=6, tz=tz)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))

def apply_y_padding(ax, s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        ax.set_ylim(0,1)
        return
    lo, hi = s.min(), s.max()
    pad = (hi - lo) * 0.08 if hi != lo else max(abs(lo)*0.02, 0.5)
    ax.set_ylim(lo - pad, hi + pad)

def et_session_to_jst_frame(base_ts_jst, session_tz, display_tz, start_hm, end_hm):
    et = base_ts_jst.tz_convert(session_tz)
    et_date = et.date()
    start_et = pd.Timestamp(et_date.year, et_date.month, et_date.day,
                             start_hm[0], start_hm[1], tz=session_tz)
    end_et = pd.Timestamp(et_date.year, et_date.month, et_date.day,
                           end_hm[0], end_hm[1], tz=session_tz)
    return start_et.tz_convert(display_tz), end_et.tz_convert(display_tz)

# ============================================================
#  描画本体
# ============================================================

def plot_df(df, key, label, mode, tz, frame=None):
    if df.empty:
        return

    # 1d の場合は陽線/陰線で色を決定
    if mode == "1d":
        open_price = df["value"].iloc[0]
        close_price = df["value"].iloc[-1]
        if close_price > open_price:
            color_line = COLOR_UP
        elif close_price < open_price:
            color_line = COLOR_DOWN
        else:
            color_line = COLOR_EQUAL
    else:
        color_line = COLOR_PRICE_DEFAULT

    fig, ax1 = plt.subplots(figsize=(9.5,4.8))
    ax1.grid(True, alpha=0.3)

    if df["volume"].abs().sum() > 0:
        ax2 = ax1.twinx()
        ax2.bar(df["time"], df["volume"], width=0.9 if mode=="1d" else 0.8,
                color=COLOR_VOLUME, alpha=0.35, zorder=1, label="Volume")

    ax1.plot(df["time"], df["value"], color=color_line, lw=2.2 if mode=="1d" else 1.8,
             solid_capstyle="round", label="Index", zorder=3)
    ax1.set_title(f"{key.upper()} ({label})", color="#ffb6c1")
    ax1.set_xlabel("Time" if mode=="1d" else "Date")
    ax1.set_ylabel("Index Value")

    format_time_axis(ax1, mode if label=="1d" else "long", tz)
    apply_y_padding(ax1, df["value"])
    if frame is not None:
        ax1.set_xlim(frame)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{key}_{label}.png", dpi=180)
    plt.close()
    log(f"saved: {key}_{label}.png color={color_line}")

# ============================================================
#  メイン処理
# ============================================================

def main():
    index_key = os.environ.get("INDEX_KEY", "").lower()
    if not index_key:
        raise SystemExit("ERROR: INDEX_KEY not set")

    MP = market_profile(index_key)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    intraday_path = find_intraday(OUTPUT_DIR, index_key)
    history_path = find_history(OUTPUT_DIR, index_key)

    intraday = read_any(intraday_path, MP["RAW_TZ_INTRADAY"], MP["DISPLAY_TZ"]) if intraday_path else pd.DataFrame()
    history = read_any(history_path, MP["RAW_TZ_HISTORY"], MP["DISPLAY_TZ"]) if history_path else pd.DataFrame()
    daily_all = to_daily(history if not history.empty else intraday, MP["DISPLAY_TZ"])

    # 1d セッション切り出し
    if not intraday.empty:
        last_ts = intraday["time"].max()
        start_jst, end_jst = et_session_to_jst_frame(last_ts,
            MP["SESSION_TZ"], MP["DISPLAY_TZ"], MP["SESSION_START"], MP["SESSION_END"])
        mask = (intraday["time"] >= start_jst) & (intraday["time"] <= end_jst)
        df_1d = intraday.loc[mask].copy()
        frame_1d = (start_jst, end_jst)
    else:
        df_1d = pd.DataFrame()
        frame_1d = None

    plot_df(df_1d, index_key, "1d", "1d", MP["DISPLAY_TZ"], frame=frame_1d)

    now = pd.Timestamp.now(tz=MP["DISPLAY_TZ"])
    for label, days in [("7d",7),("1m",31),("1y",365)]:
        sub = daily_all[daily_all["time"] >= (now - timedelta(days=days))]
        if sub.empty:
            continue
        plot_df(sub, index_key, label, "long", MP["DISPLAY_TZ"])

if __name__ == "__main__":
    main()
