#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate long-term charts (1d/7d/1m/1y).

共通ポイント
- 1d は 始値<終値: 青緑 / 始値>終値: 赤 / 同値: グレー
- 出来高があれば薄い棒で重ね描き
- 板形式CSV（銘柄ごとに列がある）もサポート：数値化できる列の等加重平均で value を作成
- 先頭列が Unnamed: 0（インデックス書き出し）でも時刻列として自動認識
- 先頭が「#」のコメント列は自動で除外
- R-BANK9 は日本株：JST 9:00–15:30 の当日セッションで 1d を切り出し

出力: docs/outputs/<index_key>_{1d|7d|1m|1y}.png
"""

from __future__ import annotations

import os
import re
from typing import Optional, Tuple, List
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# =========================
# 基本設定
# =========================

OUTPUT_DIR = "docs/outputs"

# colors
COLOR_PRICE_DEFAULT = "#ff99cc"
COLOR_VOLUME = "#7f8ca6"
COLOR_UP = "#00C2A0"   # 陽線（青緑）
COLOR_DOWN = "#FF4C4C" # 陰線（赤）
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


# =========================
# 市場セッション定義
# =========================

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

    # Astra4：米国株中心
    if k == "astra4":
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

    # fallback（JST 現物に準拠 9:00-15:00）
    return dict(
        RAW_TZ_INTRADAY="Asia/Tokyo",
        RAW_TZ_HISTORY="Asia/Tokyo",
        DISPLAY_TZ="Asia/Tokyo",
        SESSION_TZ="Asia/Tokyo",
        SESSION_START=(9, 0),
        SESSION_END=(15, 0),
    )


# =========================
# 入出力補助
# =========================

def _first(paths: List[str]) -> Optional[str]:
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def find_intraday(base: str, key: str) -> Optional[str]:
    key = key.lower()
    return _first([
        f"{base}/{key}_intraday.csv",
        f"{base}/{key}_intraday.txt",
    ])

def find_history(base: str, key: str) -> Optional[str]:
    key = key.lower()
    return _first([
        f"{base}/{key}_history.csv",
        f"{base}/{key}_history.txt",
    ])

# =========================
# 時刻/列 推定
# =========================

def ensure_tz_series(s: pd.Series, tz: str) -> pd.Series:
    if not isinstance(s, pd.Series):
        return s
    try:
        # pandas >= 2.x では .dt.tz は使えないケースがあるため安全に処理
        if pd.api.types.is_datetime64_any_dtype(s):
            if s.dt.tz is None:
                return s.dt.tz_localize(tz)
            return s.dt.tz_convert(tz)
    except Exception:
        pass
    return s

def parse_time_any(x, raw_tz: str, display_tz: str):
    """
    文字列/数値から時刻を生成。
    - UNIX秒(10桁) なら UTC → display_tz
    - それ以外は pandas に任せ、tzなしなら raw_tz を付与 → display_tz
    """
    if pd.isna(x):
        return pd.NaT
    s = str(x).strip()

    if re.fullmatch(r"\d{10}", s):
        return pd.Timestamp(int(s), unit="s", tz="UTC").tz_convert(display_tz)

    t = pd.to_datetime(s, errors="coerce")
    if pd.isna(t):
        return pd.NaT
    if getattr(t, "tzinfo", None) is None:
        try:
            t = t.tz_localize(raw_tz)
        except Exception:
            # 既に tz-aware の時があるので二重 localize を避ける
            t = t.tz_localize("UTC").tz_convert(raw_tz)
    return t.tz_convert(display_tz)

def pick_time_col_simple(cols_lower: List[str]) -> Optional[str]:
    for k in ["datetime", "time", "timestamp", "date"]:
        if k in cols_lower:
            return k
    return None

# =========================
# CSV 正規化読み込み（強化版）
# =========================

def read_any(path: Optional[str], raw_tz: str, display_tz: str) -> pd.DataFrame:
    """
    返り値: columns = [time, value, volume]

    受け付ける形式
      A) time/value[/volume] の単一列形式
      B) 板形式（time + 複数銘柄列）→ 数値化できる列を等加重平均して value を作成

    追加の堅牢化
      - Unnamed: 0 / Unnamed: <n> を時刻列として最優先採用
      - 先頭列をサンプルパースし、>=60% 成功なら時刻列とみなす
      - 先頭が '#' のコメント列は自動除外
    """
    if not path:
        return pd.DataFrame(columns=["time", "value", "volume"])

    df = pd.read_csv(path)

    raw_cols = list(df.columns)
    cols_lower = [str(c).strip().lower() for c in raw_cols]
    df.columns = cols_lower

    # コメント列除外
    drop_cols = [c for c in df.columns if str(c).strip().startswith("#")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # ---- 時刻列推定
    tcol = pick_time_col_simple(list(df.columns))

    # Unnamed: n を最優先
    if tcol is None:
        for c in df.columns:
            if re.fullmatch(r"unnamed:\s*\d+", c):
                tcol = c
                break

    # まだ無ければ先頭列を検証
    if tcol is None and len(df.columns) > 0:
        probe = df.columns[0]
        sample = df[probe].head(50).apply(lambda x: parse_time_any(x, raw_tz, display_tz))
        if sample.notna().mean() >= 0.6:
            tcol = probe

    if tcol is None:
        raise KeyError(f"No time-like column found. columns={list(df.columns)}")

    # ---- 単一 value 列形式
    def _pick_value_col(df_: pd.DataFrame) -> Optional[str]:
        cols = list(df_.columns)
        for k in ["close", "value", "index", "price", "score", "終値"]:
            if k in cols:
                return k
        nums = [c for c in cols if c != tcol and pd.api.types.is_numeric_dtype(df_[c])]
        return nums[0] if nums else None

    vcol = _pick_value_col(df)
    volcol = None
    for k in ["volume", "vol", "出来高"]:
        if k in df.columns:
            volcol = k
            break

    # A) time + value (+ volume)
    if vcol is not None and vcol in df.columns and (set(df.columns) - {tcol, vcol, volcol if volcol else ""} == set()):
        out = pd.DataFrame()
        out["time"] = df[tcol].apply(lambda x: parse_time_any(x, raw_tz, display_tz))
        out["value"] = pd.to_numeric(df[vcol], errors="coerce")
        out["volume"] = pd.to_numeric(df[volcol], errors="coerce") if volcol else 0
        out = out.dropna(subset=["time", "value"]).sort_values("time").reset_index(drop=True)
        return out

    # B) 板形式：数値化できる列を平均
    num_cols: List[str] = []
    for c in df.columns:
        if c == tcol:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() > 0:
            num_cols.append(c)

    out = pd.DataFrame()
    out["time"] = df[tcol].apply(lambda x: parse_time_any(x, raw_tz, display_tz))
    if num_cols:
        vals_df = df[num_cols].apply(lambda s: pd.to_numeric(s, errors="coerce"))
        out["value"] = vals_df.mean(axis=1)
    else:
        out["value"] = np.nan
    out["volume"] = 0

    out = out.dropna(subset=["time", "value"]).sort_values("time").reset_index(drop=True)
    return out


def to_daily(df: pd.DataFrame, display_tz: str) -> pd.DataFrame:
    """終値ベースの1日足（同一日の最終値、出来高は合算）"""
    if df.empty:
        return df
    d = df.copy()
    d["date"] = d["time"].dt.tz_convert(display_tz).dt.date
    g = d.groupby("date", as_index=False).agg({"value": "last", "volume": "sum"})
    g["time"] = pd.to_datetime(g["date"]).dt.tz_localize(display_tz)
    return g[["time", "value", "volume"]]


# =========================
# グラフ補助
# =========================

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

def session_frame_from_base(base_ts_jst: pd.Timestamp,
                            session_tz: str, display_tz: str,
                            start_hm: Tuple[int, int], end_hm: Tuple[int, int]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    et = base_ts_jst.tz_convert(session_tz)
    et_date = et.date()
    start_et = pd.Timestamp(et_date.year, et_date.month, et_date.day,
                            start_hm[0], start_hm[1], tz=session_tz)
    end_et = pd.Timestamp(et_date.year, et_date.month, et_date.day,
                          end_hm[0], end_hm[1], tz=session_tz)
    return start_et.tz_convert(display_tz), end_et.tz_convert(display_tz)


# =========================
# 描画
# =========================

def plot_df(df: pd.DataFrame, index_key: str, label: str, mode: str, tz: str,
            frame: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None):
    if df.empty:
        log(f"skip plot {index_key}_{label} (empty)")
        # 空でも軸だけ出す（デバッグ時の視認性用）
        fig, ax = plt.subplots(figsize=(9.5, 4.8))
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{index_key.upper()} ({label})", color="#ffb6c1")
        ax.set_xlabel("Time" if mode == "1d" else "Date")
        ax.set_ylabel("Index Value")
        format_time_axis(ax, mode, tz)
        outpath = f"{OUTPUT_DIR}/{index_key}_{label}.png"
        plt.tight_layout(); plt.savefig(outpath, dpi=180); plt.close()
        return

    # 1d は陽/陰/同値の色分け
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

    fig, ax1 = plt.subplots(figsize=(9.5, 4.8))
    ax1.grid(True, alpha=0.3)

    # 出来高があれば
    if "volume" in df.columns and pd.to_numeric(df["volume"], errors="coerce").fillna(0).abs().sum() > 0:
        ax2 = ax1.twinx()
        ax2.bar(df["time"], pd.to_numeric(df["volume"], errors="coerce").fillna(0),
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
    log(f"saved: {outpath}")


# =========================
# メイン
# =========================

def detect_index_key_from_outputs() -> Optional[str]:
    """docs/outputs 配下の *_intraday.csv から index_key を推定"""
    if not os.path.isdir(OUTPUT_DIR):
        return None
    for fn in os.listdir(OUTPUT_DIR):
        m = re.match(r"^([a-z0-9_+\-]+)_intraday\.csv$", fn, flags=re.I)
        if m:
            return m.group(1).lower()
    return None


def main():
    index_key = os.environ.get("INDEX_KEY", "").lower()
    if not index_key:
        index_key = detect_index_key_from_outputs() or "rbank9"  # R-BANK9 を既定

    MP = market_profile(index_key)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    intraday_path = find_intraday(OUTPUT_DIR, index_key)
    history_path = find_history(OUTPUT_DIR, index_key)

    intraday = read_any(intraday_path, MP["RAW_TZ_INTRADAY"], MP["DISPLAY_TZ"]) if intraday_path else pd.DataFrame(columns=["time","value","volume"])
    history = read_any(history_path, MP["RAW_TZ_HISTORY"], MP["DISPLAY_TZ"]) if history_path else pd.DataFrame(columns=["time","value","volume"])
    daily_all = to_daily(history if not history.empty else intraday, MP["DISPLAY_TZ"])

    # ---- 1d（セッションで切り出し）
    if not intraday.empty:
        last_ts = intraday["time"].max()
        start_jst, end_jst = session_frame_from_base(
            last_ts, MP["SESSION_TZ"], MP["DISPLAY_TZ"], MP["SESSION_START"], MP["SESSION_END"]
        )
        mask = (intraday["time"] >= start_jst) & (intraday["time"] <= end_jst)
        df_1d = intraday.loc[mask].copy()
        frame_1d = (start_jst, end_jst)
    else:
        df_1d = pd.DataFrame(columns=["time","value","volume"])
        frame_1d = None

    plot_df(df_1d, index_key, "1d", "1d", MP["DISPLAY_TZ"], frame=frame_1d)

    # ---- 7d / 1m / 1y（終値ベース）
    now = pd.Timestamp.now(tz=MP["DISPLAY_TZ"])
    for label, days in [("7d", 7), ("1m", 31), ("1y", 365)]:
        sub = daily_all[daily_all["time"] >= (now - timedelta(days=days))]
        plot_df(sub, index_key, label, "long", MP["DISPLAY_TZ"])

if __name__ == "__main__":
    main()
