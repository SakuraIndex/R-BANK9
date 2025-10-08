#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate long-term charts (1d/7d/1m/1y) for Sakura Index series.
- セッション/タイムゾーンは INDEX_KEY に応じて自動切替
- 1d は 始値<終値: 青緑 / 始値>終値: 赤 / 同値: グレー
- 値は「前日終値 = 100」へ正規化（履歴なければ当日先頭値=100でフォールバック）
- 出来高があれば薄い棒で重ね描き
出力: docs/outputs/<index_key>_{1d|7d|1m|1y}.png
"""

import os
import re
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ========================= 基本設定 =========================

OUTPUT_DIR = "docs/outputs"

# 色
COLOR_PRICE_DEFAULT = "#ff99cc"  # 長期線
COLOR_VOLUME = "#7f8ca6"
COLOR_UP = "#00C2A0"   # 陽線
COLOR_DOWN = "#FF4C4C" # 陰線
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

# ===================== 市場プロファイル =====================
def market_profile(index_key: str):
    k = (index_key or "").lower()

    # ...（米国系はそのまま）

    # S-COIN+：日本株 (JST 9:00-15:30)
    if k in ("scoin+", "scoin_plus", "scoinplus", "s-coin+"):
        return dict(
            RAW_TZ_INTRADAY="Asia/Tokyo",
            RAW_TZ_HISTORY="Asia/Tokyo",
            DISPLAY_TZ="Asia/Tokyo",
            SESSION_TZ="Asia/Tokyo",
            SESSION_START=(9, 0),
            SESSION_END=(15, 30),   # ← ここを 15:30 に
        )

    # R-BANK9：日本株 (JST 9:00-15:30)
    if k in ("rbank9", "r-bank9", "r_bank9"):
        return dict(
            RAW_TZ_INTRADAY="Asia/Tokyo",
            RAW_TZ_HISTORY="Asia/Tokyo",
            DISPLAY_TZ="Asia/Tokyo",
            SESSION_TZ="Asia/Tokyo",
            SESSION_START=(9, 0),
            SESSION_END=(15, 30),   # ← ここを 15:30 に
        )

    # fallback（日本株想定 9:00-15:30）
    return dict(
        RAW_TZ_INTRADAY="Asia/Tokyo",
        RAW_TZ_HISTORY="Asia/Tokyo",
        DISPLAY_TZ="Asia/Tokyo",
        SESSION_TZ="Asia/Tokyo",
        SESSION_START=(9, 0),
        SESSION_END=(15, 30),       # ← ここも 15:30 に
    )

# ========================== 入出力 ==========================

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
    s = str(x).strip()
    # UNIX秒
    if re.fullmatch(r"\d{10}", s):
        return pd.Timestamp(int(s), unit="s", tz="UTC").tz_convert(display_tz)
    # 汎用
    t = pd.to_datetime(s, errors="coerce")
    if pd.isna(t):
        return pd.NaT
    if t.tzinfo is None:
        t = t.tz_localize(raw_tz)
    return t.tz_convert(display_tz)

def pick_value_col(df):
    cols = [c.lower() for c in df.columns]
    for k in ["close", "price", "value", "index", "終値"]:
        if k in cols:
            return df.columns[cols.index(k)]
    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return nums[0] if nums else df.columns[0]

def pick_volume_col(df):
    cols = [c.lower() for c in df.columns]
    for k in ["volume", "vol", "出来高"]:
        if k in cols:
            return df.columns[cols.index(k)]
    return None

def read_any(path, raw_tz, display_tz):
    """列名を正規化してから時刻/値/出来高を抽出。"""
    if not path:
        return pd.DataFrame(columns=["time", "value", "volume"])

    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    # 時刻候補
    tcol = None
    for name in ["datetime", "time", "timestamp", "date"]:
        if name in df.columns:
            tcol = name
            break
    if tcol is None:
        fuzzy = [c for c in df.columns if ("time" in c) or ("date" in c)]
        if fuzzy:
            tcol = fuzzy[0]
    if tcol is None:
        raise KeyError(f"No time-like column found. columns={list(df.columns)}")

    vcol = pick_value_col(df)
    volcol = pick_volume_col(df)

    out = pd.DataFrame()
    out["time"] = df[tcol].apply(lambda x: parse_time_any(x, raw_tz, display_tz))
    out["value"] = pd.to_numeric(df[vcol], errors="coerce")
    out["volume"] = pd.to_numeric(df[volcol], errors="coerce") if volcol else 0
    out = out.dropna(subset=["time", "value"]).sort_values("time").reset_index(drop=True)
    return out

def to_daily(df, display_tz):
    if df.empty:
        return df
    d = df.copy()
    d["date"] = d["time"].dt.tz_convert(display_tz).dt.date
    g = d.groupby("date", as_index=False).agg({"value": "last", "volume": "sum"})
    g["time"] = pd.to_datetime(g["date"]).dt.tz_localize(display_tz)
    return g[["time", "value", "volume"]]

# ======================== 正規化（=100） ========================

def compute_base_level(history_df: pd.DataFrame, intraday_df: pd.DataFrame, display_tz: str) -> float:
    """
    前日終値を 100 とする基準値を返す。
    - 履歴があれば「直近営業日の終値」を基準
    - 無ければ当日の intraday の最初値を基準
    """
    # 履歴優先
    if history_df is not None and not history_df.empty:
        h = history_df.sort_values("time")
        # 直近日のさらに前日終値（＝「当日から見た直近確定日」）を優先
        # もし1行しか無ければその値を基準にする
        base_series = h["value"].dropna()
        if not base_series.empty:
            return float(base_series.iloc[-1])

    # 履歴が無い/空 → intraday の先頭
    if intraday_df is not None and not intraday_df.empty:
        first = intraday_df.sort_values("time")["value"].dropna()
        if not first.empty:
            return float(first.iloc[0])

    # 最終手段
    return 1.0

def normalize_to_100(df: pd.DataFrame, base_level: float) -> pd.DataFrame:
    out = df.copy()
    if base_level is None or not np.isfinite(base_level) or base_level == 0:
        base_level = 1.0
    out["value"] = 100.0 * (out["value"] / float(base_level))
    return out

# =========================  グラフ補助  =========================

def format_time_axis(ax, mode, tz):
    if mode == "1d":
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1, tz=tz))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tz))
    else:
        loc = mdates.AutoDateLocator(minticks=3, maxticks=6, tz=tz)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))

def apply_y_padding(ax, series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        ax.set_ylim(0, 1)
        return
    lo, hi = s.min(), s.max()
    pad = (hi - lo) * 0.08 if hi != lo else max(abs(lo) * 0.02, 0.5)
    ax.set_ylim(lo - pad, hi + pad)

def session_frame_from_base(base_ts_jst, session_tz, display_tz, start_hm, end_hm):
    # base_ts_jst をセッションタイムゾーンに変換し、その日のセッション枠をJSTに戻す
    stz = session_tz
    et = base_ts_jst.tz_convert(stz)
    d = et.date()
    start = pd.Timestamp(d.year, d.month, d.day, start_hm[0], start_hm[1], tz=stz)
    end = pd.Timestamp(d.year, d.month, d.day, end_hm[0], end_hm[1], tz=stz)
    return start.tz_convert(display_tz), end.tz_convert(display_tz)

# =========================  描画本体  =========================

def plot_df(df, index_key, label, mode, tz, frame=None):
    fig, ax1 = plt.subplots(figsize=(9.5, 4.8))
    ax1.grid(True, alpha=0.3)

    if df.empty:
        # 軸やタイトルは出す（空PNG対策）
        ax1.text(0.5, 0.5, "No data", transform=ax1.transAxes,
                 ha="center", va="center", color="#9aa6c7", fontsize=28, alpha=0.9)
        ax1.set_title(f"{index_key.upper()} ({label})", color="#ffb6c1")
        ax1.set_xlabel("Time" if mode == "1d" else "Date")
        ax1.set_ylabel("Index Value")
        format_time_axis(ax1, mode, tz)
        if frame is not None:
            ax1.set_xlim(frame)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        outpath = f"{OUTPUT_DIR}/{index_key}_{label}.png"
        plt.tight_layout()
        plt.savefig(outpath, dpi=180)
        plt.close()
        log(f"saved (empty): {outpath}")
        return

    # 1d の色
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
    if df["volume"].abs().sum() > 0:
        ax2 = ax1.twinx()
        ax2.bar(df["time"], df["volume"],
                width=0.9 if mode == "1d" else 0.8,
                color=COLOR_VOLUME, alpha=0.30, zorder=1, label="Volume")

    ax1.plot(df["time"], df["value"], color=color_line, lw=lw,
             solid_capstyle="round", label="Index", zorder=3)
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

# ============================ メイン ============================

def main():
    index_key = os.environ.get("INDEX_KEY", "").lower()
    if not index_key:
        raise SystemExit("ERROR: INDEX_KEY not set")

    MP = market_profile(index_key)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    intraday_path = find_intraday(OUTPUT_DIR, index_key)
    history_path = find_history(OUTPUT_DIR, index_key)

    intraday_raw = read_any(intraday_path, MP["RAW_TZ_INTRADAY"], MP["DISPLAY_TZ"]) if intraday_path else pd.DataFrame()
    history_raw = read_any(history_path, MP["RAW_TZ_HISTORY"], MP["DISPLAY_TZ"]) if history_path else pd.DataFrame()

    # 基準値（前日終値 or 当日先頭値）
    base = compute_base_level(history_raw, intraday_raw, MP["DISPLAY_TZ"])
    intraday = normalize_to_100(intraday_raw, base) if not intraday_raw.empty else intraday_raw
    history = normalize_to_100(history_raw, base) if not history_raw.empty else history_raw

    # 日次系列（終値ベース）
    daily_all = to_daily(history if not history.empty else intraday, MP["DISPLAY_TZ"])

    # ---- 1d（セッション切り出し）----
    if not intraday.empty:
        last_ts = intraday["time"].max()
        start_jst, end_jst = session_frame_from_base(
            last_ts, MP["SESSION_TZ"], MP["DISPLAY_TZ"],
            MP["SESSION_START"], MP["SESSION_END"]
        )
        mask = (intraday["time"] >= start_jst) & (intraday["time"] <= end_jst)
        df_1d = intraday.loc[mask].copy()
        # 1d はギャップ対策でソートと重複除去
        df_1d = df_1d.sort_values("time").drop_duplicates(subset="time")
        frame_1d = (start_jst, end_jst)
    else:
        df_1d = pd.DataFrame(columns=["time", "value", "volume"])
        frame_1d = None

    plot_df(df_1d, index_key, "1d", "1d", MP["DISPLAY_TZ"], frame=frame_1d)

    # ---- 7d / 1m / 1y（終値ベースの長期）----
    now = pd.Timestamp.now(tz=MP["DISPLAY_TZ"])
    for label, days in [("7d", 7), ("1m", 31), ("1y", 365)]:
        sub = daily_all[daily_all["time"] >= (now - timedelta(days=days))] if not daily_all.empty else pd.DataFrame()
        sub = sub.sort_values("time").drop_duplicates(subset="time")
        plot_df(sub, index_key, label, "long", MP["DISPLAY_TZ"])

if __name__ == "__main__":
    main()
