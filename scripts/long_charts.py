#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate long-term charts (1d / 7d / 1m / 1y) for Sakura Index series.

- セッション/タイムゾーンは INDEX_KEY に応じて自動切替
- 1d は 始値<終値: 青緑 / 始値>終値: 赤 / 同値: グレー
- 出来高があれば薄い棒で重ね描き
- 行方向＝時刻、列方向＝銘柄（数値列）という板状CSVにも対応（行平均で等加重）

出力: docs/outputs/<index_key>_{1d|7d|1m|1y}.png
必要: 環境変数 INDEX_KEY（例: rbank9, scoin_plus, ain10 など）
"""

import os
import re
from datetime import timedelta
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ============================================================
#  基本設定
# ============================================================

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


# ============================================================
#  市場セッション定義（INDEX_KEY で切替）
# ============================================================

def market_profile(index_key: str):
    k = (index_key or "").lower()

    # AIN-10：米国株 (ET 9:30-16:00 → JST表示)
    if k == "ain10":
        return dict(
            RAW_TZ_INTRADAY="America/New_York",
            RAW_TZ_HISTORY="Asia/Tokyo",
            DISPLAY_TZ="Asia/Tokyo",
            SESSION_TZ="America/New_York",
            SESSION_START=(9, 30),
            SESSION_END=(16, 0),
        )

    # S-COIN+：日本株 (JST 9:00-15:30)
    if k in ("scoin+", "scoin_plus", "scoinplus", "s-coin+"):
        return dict(
            RAW_TZ_INTRADAY="Asia/Tokyo",
            RAW_TZ_HISTORY="Asia/Tokyo",
            DISPLAY_TZ="Asia/Tokyo",
            SESSION_TZ="Asia/Tokyo",
            SESSION_START=(9, 0),
            SESSION_END=(15, 30),
        )

    # R-BANK9：日本株 (JST 9:00-15:30)
    if k in ("rbank9", "r-bank9", "r_bank9"):
        return dict(
            RAW_TZ_INTRADAY="Asia/Tokyo",
            RAW_TZ_HISTORY="Asia/Tokyo",
            DISPLAY_TZ="Asia/Tokyo",
            SESSION_TZ="Asia/Tokyo",
            SESSION_START=(9, 0),
            SESSION_END=(15, 30),
        )

    # Astra4（例）：米国株 (ET 9:30-16:00 → JST表示)
    if k == "astra4":
        return dict(
            RAW_TZ_INTRADAY="America/New_York",
            RAW_TZ_HISTORY="Asia/Tokyo",
            DISPLAY_TZ="Asia/Tokyo",
            SESSION_TZ="America/New_York",
            SESSION_START=(9, 30),
            SESSION_END=(16, 0),
        )

    # fallback（JST 現物に準拠）
    return dict(
        RAW_TZ_INTRADAY="Asia/Tokyo",
        RAW_TZ_HISTORY="Asia/Tokyo",
        DISPLAY_TZ="Asia/Tokyo",
        SESSION_TZ="Asia/Tokyo",
        SESSION_START=(9, 0),
        SESSION_END=(15, 0),
    )


# ============================================================
#  入出力ユーティリティ
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

def ensure_tz(series: pd.Series, tz: str) -> pd.Series:
    """Series を指定 tz の tz-aware に統一（naive→localize, 他 tz→convert）。"""
    s = pd.to_datetime(series, errors="coerce")
    try:
        if getattr(s.dt, "tz", None) is None:
            s = s.dt.tz_localize(tz)
        else:
            s = s.dt.tz_convert(tz)
    except Exception:
        def _fix(x):
            if pd.isna(x):
                return pd.NaT
            x = pd.to_datetime(x, errors="coerce")
            if pd.isna(x):
                return pd.NaT
            if x.tzinfo is None:
                return x.tz_localize(tz)
            return x.tz_convert(tz)
        s = s.apply(_fix)
    return s

def parse_time_any(x, raw_tz: str, display_tz: str):
    if pd.isna(x):
        return pd.NaT
    s = str(x).strip()
    # UNIX秒対応
    if re.fullmatch(r"\d{10}", s):
        return pd.Timestamp(int(s), unit="s", tz="UTC").tz_convert(display_tz)
    # 汎用
    try:
        t = pd.to_datetime(s, errors="coerce")
        if pd.isna(t):
            return pd.NaT
        if getattr(t, "tzinfo", None) is None:
            t = t.tz_localize(raw_tz)
        return t.tz_convert(display_tz)
    except Exception:
        return pd.NaT

def pick_time_col(cols_lower: List[str]) -> Optional[str]:
    candidates = ["datetime", "time", "timestamp", "date", "unnamed: 0"]
    for name in candidates:
        if name in cols_lower:
            return name
    fuzzy = [i for i, c in enumerate(cols_lower) if ("time" in c) or ("date" in c)]
    return cols_lower[fuzzy[0]] if fuzzy else None

    # 盤面形式: 時刻列以外の「数値列」を平均
    num_cols = []
    for c in df.columns:
        if c == tcol:
            continue
        # 文字列混在でも数値化できれば対象（coerce）
        as_num = pd.to_numeric(df[c], errors="coerce")
        if as_num.notna().sum() > 0:
            num_cols.append(c)

    if len(num_cols) == 0:
        # すべて非数値なら空
        return pd.DataFrame(columns=["time", "value", "volume"])

    out = pd.DataFrame()
    out["time"] = df[tcol].apply(lambda x: parse_time_any(x, raw_tz, display_tz))
    out["time"] = ensure_tz(out["time"], display_tz)

    # 🔽 ここを修正（2行まとめて置き換え）
    vals_df = df[num_cols].apply(lambda s: pd.to_numeric(s, errors="coerce"))
    out["value"] = vals_df.mean(axis=1)

    out["volume"] = 0  # 盤面からは出来高なし
    out = out.dropna(subset=["time", "value"]).sort_values("time").reset_index(drop=True)
    return out

    # 盤面形式: 時刻列以外の「数値列」を平均
    num_cols = []
    for c in df.columns:
        if c == tcol:
            continue
        # 文字列混在でも数値化できれば対象（coerce）
        as_num = pd.to_numeric(df[c], errors="coerce")
        if as_num.notna().sum() > 0:
            num_cols.append(c)

    if len(num_cols) == 0:
        # すべて非数値なら空
        return pd.DataFrame(columns=["time", "value", "volume"])

    out = pd.DataFrame()
    out["time"] = df[tcol].apply(lambda x: parse_time_any(x, raw_tz, display_tz))
    out["time"] = ensure_tz(out["time"], display_tz)
    # 等加重平均（列方向の平均）
    out["value"] = pd.to_numeric(df[num_cols], errors="coerce").mean(axis=1)
    out["volume"] = 0  # 盤面からは出来高なし
    out = out.dropna(subset=["time", "value"]).sort_values("time").reset_index(drop=True)
    return out

def to_daily(df: pd.DataFrame, display_tz: str) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.copy()
    d["time"] = ensure_tz(d["time"], display_tz)
    d["date"] = d["time"].dt.date
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

def session_frame(base_ts_jst: pd.Timestamp, session_tz: str, display_tz: str,
                  start_hm: tuple, end_hm: tuple):
    stz = session_tz
    # 「その日のセッション」を session_tz で求め、表示tzへ変換
    base_in_sess = base_ts_jst.tz_convert(stz)
    d = base_in_sess.date()
    start_sess = pd.Timestamp(d.year, d.month, d.day, start_hm[0], start_hm[1], tz=stz)
    end_sess   = pd.Timestamp(d.year, d.month, d.day, end_hm[0], end_hm[1], tz=stz)
    return start_sess.tz_convert(display_tz), end_sess.tz_convert(display_tz)


# ============================================================
#  描画本体
# ============================================================

def plot_df(df: pd.DataFrame, index_key: str, label: str, mode: str, tz: str,
            frame=None):
    fig, ax1 = plt.subplots(figsize=(9.5, 4.8))
    ax1.grid(True, alpha=0.3)

    if df is None or df.empty:
        # 空データのプレースホルダー
        ax1.set_title(f"{index_key.upper()} ({label})", color="#ffb6c1")
        ax1.set_xlabel("Time" if mode == "1d" else "Date")
        ax1.set_ylabel("Index Value")
        format_time_axis(ax1, mode, tz)
        if frame is not None:
            ax1.set_xlim(frame)
        ax1.text(0.5, 0.5, "No data", transform=ax1.transAxes,
                 ha="center", va="center", color="#b8c2e0", fontsize=18, alpha=0.7)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        outpath = f"{OUTPUT_DIR}/{index_key}_{label}.png"
        plt.tight_layout()
        plt.savefig(outpath, dpi=180)
        plt.close()
        log(f"saved empty: {outpath}")
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
    if "volume" in df.columns and pd.to_numeric(df["volume"], errors="coerce").fillna(0).abs().sum() > 0:
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
    log(f"saved: {outpath} (mode={mode})")


# ============================================================
#  メイン
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
    history  = read_any(history_path,  MP["RAW_TZ_HISTORY"],  MP["DISPLAY_TZ"]) if history_path  else pd.DataFrame()

    # 念押しで tz を統一
    if not intraday.empty:
        intraday["time"] = ensure_tz(intraday["time"], MP["DISPLAY_TZ"])
    if not history.empty:
        history["time"] = ensure_tz(history["time"], MP["DISPLAY_TZ"])

    daily_all = to_daily(history if not history.empty else intraday, MP["DISPLAY_TZ"])

    # ---- 1d（セッションで切り出し）----
    if not intraday.empty:
        last_ts = intraday["time"].max()
        start_jst, end_jst = session_frame(
            last_ts, MP["SESSION_TZ"], MP["DISPLAY_TZ"],
            MP["SESSION_START"], MP["SESSION_END"]
        )
        # x 軸を必ずセッション枠に
        mask = (intraday["time"] >= start_jst) & (intraday["time"] <= end_jst)
        df_1d = intraday.loc[mask].copy()
        # データが丸ごと枠外なら空で出力（No data）
        if df_1d.empty and not intraday.empty:
            df_1d = pd.DataFrame()  # 空
        frame_1d = (start_jst, end_jst)
    else:
        df_1d = pd.DataFrame()
        frame_1d = None

    plot_df(df_1d, index_key, "1d", "1d", MP["DISPLAY_TZ"], frame=frame_1d)

    # ---- 7d / 1m / 1y（終値ベースの長期）----
    now = pd.Timestamp.now(tz=MP["DISPLAY_TZ"])
    for label, days in [("7d", 7), ("1m", 31), ("1y", 365)]:
        sub = daily_all[daily_all["time"] >= (now - timedelta(days=days))]
        plot_df(sub if not sub.empty else pd.DataFrame(),
                index_key, label, "long", MP["DISPLAY_TZ"])

if __name__ == "__main__":
    main()
