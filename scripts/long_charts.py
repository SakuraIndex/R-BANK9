#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
R-BANK9 long-term charts generator (1d / 7d / 1m / 1y)
- 日本株（JST 9:00〜15:30）
- 板形式CSV（時刻+複数銘柄列）/ 単一列CSV（time,value[,volume]）の両対応
- 履歴があればベース100に正規化し、1d は前日終値から連続になるようスケール
- 履歴が無い場合でも 1d は描画、長期はフォールバックで当日1点だけ描画
"""

import os
import re
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

OUTPUT_DIR = "docs/outputs"
INDEX_KEY = "rbank9"  # このリポ専用

# colors
C_BG = "#0b0f1a"
C_GRID = "#27314a"
C_FG = "#e5ecff"
C_TICK = "#b8c2e0"
C_LINE_D = "#00C2A0"   # 1d 線（陽線/陰線の代わりに固定色で）
C_LINE_L = "#ff99cc"   # 長期線

plt.rcParams.update({
    "figure.facecolor": C_BG,
    "axes.facecolor":   C_BG,
    "axes.edgecolor":   C_GRID,
    "axes.labelcolor":  C_FG,
    "xtick.color":      C_TICK,
    "ytick.color":      C_TICK,
    "grid.color":       C_GRID,
    "font.family":      "Noto Sans CJK JP",
})

JST = "Asia/Tokyo"

# ───────────── ユーティリティ ─────────────

def log(msg: str):
    print(f"[long_charts] {msg}")

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

def parse_time_any(x, raw_tz: str, display_tz: str):
    """文字列/数値を Timestamp(tz=display_tz) に"""
    if pd.isna(x):
        return pd.NaT
    s = str(x).strip()

    # UNIX秒
    if re.fullmatch(r"\d{10}", s):
        return pd.Timestamp(int(s), unit="s", tz="UTC").tz_convert(display_tz)

    t = pd.to_datetime(s, errors="coerce", utc=False)
    if pd.isna(t):
        return pd.NaT
    if t.tzinfo is None:
        t = t.tz_localize(raw_tz)
    return t.tz_convert(display_tz)

def pick_time_col(cols_lower):
    for k in ["datetime", "time", "timestamp", "date"]:
        if k in cols_lower:
            return k
    fuzzy = [c for c in cols_lower if ("time" in c) or ("date" in c)]
    return fuzzy[0] if fuzzy else None

def pick_value_col(df):
    cols = [c.lower() for c in df.columns]
    for k in ["close", "value", "index", "price", "score", "終値"]:
        if k in cols:
            return df.columns[cols.index(k)]
    # 数値列の先頭
    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return nums[0] if nums else None

def pick_volume_col(df):
    cols = [c.lower() for c in df.columns]
    for k in ["volume", "vol", "出来高"]:
        if k in cols:
            return df.columns[cols.index(k)]
    return None

def read_any(path: Optional[str], raw_tz: str, display_tz: str) -> pd.DataFrame:
    """
    返り値: time,value,volume
    - 形式A: time / value [/ volume]
    - 形式B: time / <銘柄列...> を数値化 → 等加重平均で value
    """
    if not path:
        return pd.DataFrame(columns=["time", "value", "volume"])

    df = pd.read_csv(path)
    raw_cols = list(df.columns)
    df.columns = [str(c).strip().lower() for c in raw_cols]

    tcol = pick_time_col(list(df.columns))
    if tcol is None:
        raise KeyError(f"No time-like column found. columns={list(df.columns)}")

    vcol = pick_value_col(df)
    volcol = pick_volume_col(df)

    # 形式A: 単一 value
    if vcol and vcol in df.columns and (set(df.columns) - {tcol, vcol, volcol if volcol else ""} == set()):
        out = pd.DataFrame()
        out["time"] = df[tcol].apply(lambda x: parse_time_any(x, raw_tz, display_tz))
        out["value"] = pd.to_numeric(df[vcol], errors="coerce")
        out["volume"] = pd.to_numeric(df[volcol], errors="coerce") if volcol else 0
        return out.dropna(subset=["time", "value"]).sort_values("time").reset_index(drop=True)

    # 形式B: 板形式（time 以外の数値列平均）
    num_cols = []
    for c in df.columns:
        if c == tcol: 
            continue
        as_num = pd.to_numeric(df[c], errors="coerce")
        if as_num.notna().sum() > 0:
            num_cols.append(c)

    out = pd.DataFrame()
    out["time"] = df[tcol].apply(lambda x: parse_time_any(x, raw_tz, display_tz))
    if len(num_cols) == 0:
        out["value"] = np.nan
    else:
        vals = df[num_cols].apply(lambda s: pd.to_numeric(s, errors="coerce"))
        out["value"] = vals.mean(axis=1)

    out["volume"] = 0
    return out.dropna(subset=["time", "value"]).sort_values("time").reset_index(drop=True)

def to_daily(df: pd.DataFrame, display_tz: str) -> pd.DataFrame:
    """time,value,volume → 日次終値（最終値）"""
    if df.empty:
        return df
    d = df.copy()
    d["date"] = d["time"].dt.tz_convert(display_tz).dt.date
    g = d.groupby("date", as_index=False).agg({"value": "last", "volume": "sum"})
    g["time"] = pd.to_datetime(g["date"]).dt.tz_localize(display_tz)
    return g[["time", "value", "volume"]]

def normalize_base100(daily: pd.DataFrame) -> pd.DataFrame:
    """最初の値を100に正規化"""
    if daily.empty:
        return daily
    base = daily["value"].iloc[0]
    if pd.isna(base) or base == 0:
        return daily
    out = daily.copy()
    out["value"] = out["value"] / base * 100.0
    return out

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
    pad = (hi - lo) * 0.08 if hi != lo else max(abs(lo) * 0.02, 0.2)
    ax.set_ylim(lo - pad, hi + pad)

def jst_session_frame(base_ts_jst: pd.Timestamp):
    """当日 9:00–15:30 の xlim"""
    d = base_ts_jst.tz_convert(JST).date()
    start = pd.Timestamp(d.year, d.month, d.day, 9, 0, tz=JST)
    end   = pd.Timestamp(d.year, d.month, d.day, 15, 30, tz=JST)
    return start, end

# ───────────── 描画 ─────────────

def plot_df(df, title_key, label, mode, tz, frame=None, color=C_LINE_L):
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.grid(True, alpha=0.3)
    if df.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color="#9aa3b2", fontsize=22)
    else:
        ax.plot(df["time"], df["value"], color=color, lw=2.0,
                solid_capstyle="round", label="Index", zorder=3)
        apply_y_padding(ax, df["value"])
        if frame is not None:
            ax.set_xlim(frame)
    ax.set_title(f"{title_key.upper()} ({label})", color="#ffb6c1")
    ax.set_xlabel("Time" if mode == "1d" else "Date")
    ax.set_ylabel("Index Value")
    format_time_axis(ax, mode, tz)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outpath = f"{OUTPUT_DIR}/{title_key}_{label}.png"
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()
    log(f"saved: {outpath}")

# ───────────── メイン ─────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    intraday_path = find_intraday(OUTPUT_DIR, INDEX_KEY)
    history_path  = find_history(OUTPUT_DIR, INDEX_KEY)

    intraday = read_any(intraday_path, JST, JST) if intraday_path else pd.DataFrame()
    history  = read_any(history_path,  JST, JST) if history_path  else pd.DataFrame()

    # 日次履歴を整形 & ベース100
    daily_hist = to_daily(history, JST)
    daily_hist = normalize_base100(daily_hist)

    # ── 1d（連続スケール）
    if not intraday.empty:
        intraday = intraday.sort_values("time")
        # 1d の相対変化（起点=最初の値）
        start_v = intraday["value"].iloc[0]
        intr_rel = intraday.copy()
        if start_v and not pd.isna(start_v) and start_v != 0:
            intr_rel["value"] = intr_rel["value"] / start_v

        # 履歴の最新終値（ベース100）に接続
        if not daily_hist.empty:
            last_close_100 = daily_hist["value"].iloc[-1]  # 既に100ベース
            intr_rel["value"] = intr_rel["value"] * last_close_100
        else:
            # 履歴無し：とりあえず 100 ベースで表示
            intr_rel["value"] = intr_rel["value"] * 100.0

        # 取引時間で切り出し
        last_ts = intraday["time"].max()
        x0, x1 = jst_session_frame(last_ts)
        mask = (intr_rel["time"] >= x0) & (intr_rel["time"] <= x1)
        df_1d = intr_rel.loc[mask].copy()
        frame = (x0, x1)
    else:
        df_1d = pd.DataFrame()
        frame = None

    plot_df(df_1d, INDEX_KEY, "1d", "1d", JST, frame=frame, color=C_LINE_D)

    # ── 7d / 1m / 1y
    now = pd.Timestamp.now(tz=JST)

    # 履歴が無い場合のフォールバック（当日だけでも置く）
    daily_all = daily_hist.copy()
    if daily_all.empty and not intraday.empty:
        d0 = intraday.iloc[-1]
        daily_all = pd.DataFrame({
            "time":   [pd.Timestamp(d0["time"]).tz_convert(JST).normalize()],
            "value":  [df_1d["value"].iloc[-1] if not df_1d.empty else np.nan],
            "volume": [0],
        })

    windows = [("7d", 7), ("1m", 31), ("1y", 365)]
    for label, days in windows:
        sub = daily_all[daily_all["time"] >= (now - timedelta(days=days))]
        plot_df(sub, INDEX_KEY, label, "long", JST, color=C_LINE_L)

if __name__ == "__main__":
    main()
