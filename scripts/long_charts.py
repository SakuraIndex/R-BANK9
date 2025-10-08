#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate long-term charts (1d/7d/1m/1y) for Sakura Index series.

- セッション/タイムゾーンは INDEX_KEY で切替
- 1d は 始値<終値: 青緑 / 始値>終値: 赤 / 同値: グレー
- 出来高があれば薄い棒で重ね描き
- データが空でも必ず PNG を保存（"No data" プレースホルダー）

出力先: docs/outputs/<index_key>_{1d|7d|1m|1y}.png
"""

import os
import re
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

OUTPUT_DIR = "docs/outputs"

# ログ（最低限）
print(f"[long_charts] cwd={os.getcwd()}")
print(f"[long_charts] OUTPUT_DIR={os.path.abspath(OUTPUT_DIR)}")

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

# ------------------------------------------------------------
# 市場プロファイル
# ------------------------------------------------------------
def market_profile(index_key: str):
    k = (index_key or "").lower()

    # AIN-10（米国株をJST表示）
    if k == "ain10":
        return dict(
            RAW_TZ_INTRADAY="America/New_York",
            RAW_TZ_HISTORY="Asia/Tokyo",
            DISPLAY_TZ="Asia/Tokyo",
            SESSION_TZ="America/New_York",
            SESSION_START=(9, 30),
            SESSION_END=(16, 0),
        )

    # S-COIN+（日本株）
    if k in ("scoin+", "scoin_plus", "scoinplus", "s-coin+"):
        return dict(
            RAW_TZ_INTRADAY="Asia/Tokyo",
            RAW_TZ_HISTORY="Asia/Tokyo",
            DISPLAY_TZ="Asia/Tokyo",
            SESSION_TZ="Asia/Tokyo",
            SESSION_START=(9, 0),
            SESSION_END=(15, 30),
        )

    # R-BANK9（日本株）※今回の対象：9:00–15:30 JST
    if k == "rbank9":
        return dict(
            RAW_TZ_INTRADAY="Asia/Tokyo",
            RAW_TZ_HISTORY="Asia/Tokyo",
            DISPLAY_TZ="Asia/Tokyo",
            SESSION_TZ="Asia/Tokyo",
            SESSION_START=(9, 0),
            SESSION_END=(15, 30),
        )

    # fallback（JST）
    return dict(
        RAW_TZ_INTRADAY="Asia/Tokyo",
        RAW_TZ_HISTORY="Asia/Tokyo",
        DISPLAY_TZ="Asia/Tokyo",
        SESSION_TZ="Asia/Tokyo",
        SESSION_START=(9, 0),
        SESSION_END=(15, 0),
    )

# ------------------------------------------------------------
# 入出力ユーティリティ
# ------------------------------------------------------------
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
    try:
        t = pd.to_datetime(s, errors="coerce")
        if pd.isna(t):
            return pd.NaT
        if t.tzinfo is None:
            t = t.tz_localize(raw_tz)
        return t.tz_convert(display_tz)
    except Exception:
        return pd.NaT

# ---- 頑丈版 read_any（コメント・Unnamed列・横持ち＝等加重対応）----

def read_any(path, raw_tz, display_tz):
    """
    CSV/TXT から intraday/history を読む。
    - 先頭が '#' の行はコメント無視
    - 時刻列を推定
    - 値列が無ければ数値列の等加重平均
    - 出来高はあれば採用
    """
    if not path:
        return pd.DataFrame(columns=["time", "value", "volume"])

    df = pd.read_csv(path, comment="#", skip_blank_lines=True, engine="python")
    df.columns = [str(c).strip().lower() for c in df.columns]

    # 時刻列推定
    tcol = None
    for c in ["datetime", "time", "timestamp", "date"]:
        if c in df.columns:
            tcol = c
            break
    if tcol is None and "unnamed: 0" in df.columns:
        tcol = "unnamed: 0"
    if tcol is None:
        fuzzy = [c for c in df.columns if ("time" in c) or ("date" in c)]
        if fuzzy:
            tcol = fuzzy[0]
    if tcol is None:
        tcol = df.columns[0]

    # 値/出来高候補
    vcol = next((c for c in ["close", "price", "value", "index", "終値"] if c in df.columns), None)
    volcol = next((c for c in ["volume", "vol", "出来高"] if c in df.columns), None)

    out = pd.DataFrame()
    out["time"] = df[tcol].apply(lambda x: parse_time_any(x, raw_tz, display_tz))

    if vcol is not None:
        out["value"] = pd.to_numeric(df[vcol], errors="coerce")
    else:
        # 代表列が無い → 数値列の等加重平均
        num_cols = [c for c in df.columns if c != tcol]
        if len(num_cols) == 0:
            out["value"] = np.nan
        else:
            # 各列ごとに to_numeric（DataFrame まとめては不可）
            vals = df[num_cols].apply(lambda s: pd.to_numeric(s, errors="coerce"))
            # すべて NaN になった列は無視（全部NaNだと mean が NaN のままなのでOK）
            out["value"] = vals.mean(axis=1)

    out["volume"] = pd.to_numeric(df[volcol], errors="coerce").fillna(0) if volcol else 0
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

# ------------------------------------------------------------
# グラフ補助
# ------------------------------------------------------------
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

def et_session_to_jst_frame(base_ts_jst, session_tz, display_tz, start_hm, end_hm):
    et = base_ts_jst.tz_convert(session_tz)
    et_date = et.date()
    start_et = pd.Timestamp(et_date.year, et_date.month, et_date.day,
                            start_hm[0], start_hm[1], tz=session_tz)
    end_et = pd.Timestamp(et_date.year, et_date.month, et_date.day,
                          end_hm[0], end_hm[1], tz=session_tz)
    return start_et.tz_convert(display_tz), end_et.tz_convert(display_tz)

# ------------------------------------------------------------
# 描画（空でも必ず保存）
# ------------------------------------------------------------
def plot_df(df, index_key, label, mode, tz, frame=None):
    fig, ax1 = plt.subplots(figsize=(9.5, 4.8))
    ax1.grid(True, alpha=0.3)

    if df is None or df.empty:
        ax1.set_title(f"{index_key.upper()} ({label})", color="#ffb6c1")
        ax1.set_xlabel("Time" if mode == "1d" else "Date")
        ax1.set_ylabel("Index Value")
        format_time_axis(ax1, mode, tz)
        if frame is not None:
            ax1.set_xlim(frame)
        ax1.text(0.5, 0.5, "No data", transform=ax1.transAxes,
                 ha="center", va="center", color="#b8c2e0", fontsize=24, alpha=0.8)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        outpath = f"{OUTPUT_DIR}/{index_key}_{label}.png"
        plt.tight_layout()
        plt.savefig(outpath, dpi=180)
        plt.close()
        print(f"[long_charts] saved (placeholder): {os.path.abspath(outpath)}  exists={os.path.exists(outpath)}")
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

    # 出来高
    if df["volume"].abs().sum() > 0:
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

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outpath = f"{OUTPUT_DIR}/{index_key}_{label}.png"
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()
    print(f"[long_charts] saved: {os.path.abspath(outpath)}  exists={os.path.exists(outpath)}")

# ------------------------------------------------------------
# メイン
# ------------------------------------------------------------
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

    # 1d（セッション切り出し）
    if not intraday.empty:
        last_ts = intraday["time"].max()
        # セッションはプロファイル依存（R-BANK9 は JST 9:00–15:30）
        start_jst = pd.Timestamp(last_ts.year, last_ts.month, last_ts.day,
                                 MP["SESSION_START"][0], MP["SESSION_START"][1],
                                 tz=MP["DISPLAY_TZ"])
        end_jst   = pd.Timestamp(last_ts.year, last_ts.month, last_ts.day,
                                 MP["SESSION_END"][0], MP["SESSION_END"][1],
                                 tz=MP["DISPLAY_TZ"])
        mask = (intraday["time"] >= start_jst) & (intraday["time"] <= end_jst)
        df_1d = intraday.loc[mask].copy()
        frame_1d = (start_jst, end_jst)
    else:
        df_1d = pd.DataFrame()
        frame_1d = None

    plot_df(df_1d, index_key, "1d", "1d", MP["DISPLAY_TZ"], frame=frame_1d)

    # 7d / 1m / 1y（終値ベース）
    now = pd.Timestamp.now(tz=MP["DISPLAY_TZ"])
    for label, days in [("7d", 7), ("1m", 31), ("1y", 365)]:
        sub = daily_all[daily_all["time"] >= (now - timedelta(days=days))]
        plot_df(sub, index_key, label, "long", MP["DISPLAY_TZ"])

if __name__ == "__main__":
    main()
