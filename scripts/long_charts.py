#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate R-BANK9 long-term charts (1d / 7d / 1m / 1y)

- 1d:
  * 日本株の当日セッション (JST 09:00–15:30) のみ表示
  * 板形式CSV（銘柄が横並び）/ 単一列形式（time,value[,volume]）の両方を自動解釈
  * 等加重平均で value を算出
  * ダークテーマで軸・ラベル・グリッドを明示

- 7d/1m/1y:
  * docs/outputs/<index>_history.csv（date,value）を使用
  * データが1点以下でも「No data」注記を入れてPNGを保存
  * 背景色・スタイルを1dと統一

出力:
  docs/outputs/<index>_1d.png
  docs/outputs/<index>_7d.png
  docs/outputs/<index>_1m.png
  docs/outputs/<index>_1y.png
"""

from __future__ import annotations

import os
import re
from typing import Optional, List

import pandas as pd
import numpy as np
import pytz
import matplotlib
import matplotlib.pyplot as plt

# ====== 定数 ======
JP_TZ = "Asia/Tokyo"
SESSION_START = "09:00"
SESSION_END   = "15:30"

# ダークテーマ色
BG = "#0E1117"
FG = "#E6E6E6"
ACCENT = "#3bd6c6"   # ティール系ライン
TITLE  = "#f2b6c6"   # ピンク系タイトル
GRID_A = 0.25

# ====== Matplotlib 全体設定 ======
matplotlib.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "axes.edgecolor": FG,
    "axes.labelcolor": FG,
    "xtick.color": FG,
    "ytick.color": FG,
    "text.color": FG,
    "grid.color": FG,
    "savefig.facecolor": BG,
})


# ====== ユーティリティ ======
def _lower_cols(df: pd.DataFrame) -> List[str]:
    raw = list(df.columns)
    cols_lower = [str(c).strip().lower() for c in raw]
    df.columns = cols_lower
    return cols_lower


def _pick_time_col(cols_lower: List[str]) -> Optional[str]:
    for k in ("time", "timestamp", "date", "datetime"):
        if k in cols_lower:
            return k
    # Unnamed: 0 を time 扱い
    for c in cols_lower:
        if c.startswith("unnamed") and ": 0" in c:
            return c
    # あいまい一致
    for c in cols_lower:
        if ("time" in c) or ("date" in c):
            return c
    return None


def _ensure_tz(series: pd.Series, tz: str) -> pd.Series:
    if not isinstance(series, pd.Series):
        return series
    if getattr(series.dt, "tz", None) is None:
        # 値がUTC/naive混在でも to_datetime(utc=True) 経由で吸収
        dt = pd.to_datetime(series, utc=True, errors="coerce")
    else:
        # 既に tz-aware
        dt = series.dt.tz_convert("UTC")
    return dt.dt.tz_convert(tz)


def parse_time_any(x, raw_tz: str, display_tz: str):
    try:
        ts = pd.to_datetime(x, utc=True)
    except Exception:
        ts = pd.NaT
    if pd.isna(ts):
        return pd.NaT
    if ts.tzinfo is None:
        # もしUTC基準で来た場合は raw_tz とみなしてローカライズ
        ts = ts.tz_localize(raw_tz)
    return ts.tz_convert(display_tz)


def read_any_intraday(path: Optional[str], raw_tz: str, display_tz: str) -> pd.DataFrame:
    """
    intraday CSV を読み込み、必ず ["time","value","volume"] を返す。
    - time/value/volume 形式 or 銘柄横並び形式（等加重平均）
    - 先頭 "Unnamed: 0" も time として扱う
    - コメント列（先頭が "#"）は無視
    """
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=["time", "value", "volume"])

    df = pd.read_csv(path, dtype=str)  # 一旦文字列で読み込み → 数値化は後で
    # コメント列除外（"# 〜"）: そのまま入っていたら削除
    drop_cols = [c for c in df.columns if str(c).strip().startswith("#")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    cols_lower = _lower_cols(df)
    tcol = _pick_time_col(cols_lower)
    if tcol is None:
        raise KeyError(f"No time-like column found. columns={list(df.columns)}")

    # 候補: 単一の value/volume
    vcol = None
    volcol = None
    for c in df.columns:
        lc = c.lower()
        if lc in ("value", "index", "score") or ("value" in lc):
            vcol = c
        if lc == "volume" or ("volume" in lc):
            volcol = c

    out = pd.DataFrame()
    out["time"] = df[tcol].apply(lambda x: parse_time_any(x, raw_tz, display_tz))
    out["time"] = _ensure_tz(out["time"], display_tz)

    if vcol is not None:
        out["value"] = pd.to_numeric(df[vcol], errors="coerce")
        out["volume"] = pd.to_numeric(df[volcol], errors="coerce") if (volcol and volcol in df.columns) else 0
    else:
        # 板形式：time 以外を数値化 → 等加重平均
        num_cols: List[str] = []
        for c in df.columns:
            if c == tcol:
                continue
            as_num = pd.to_numeric(df[c], errors="coerce")
            if as_num.notna().sum() > 0:
                num_cols.append(c)
        if len(num_cols) == 0:
            return pd.DataFrame(columns=["time", "value", "volume"])
        vals_df = df[num_cols].apply(lambda s: pd.to_numeric(s, errors="coerce"))
        out["value"] = vals_df.mean(axis=1)
        out["volume"] = 0

    out = out.dropna(subset=["time", "value"]).sort_values("time").reset_index(drop=True)
    return out


def clamp_to_session_today_jst(df: pd.DataFrame) -> pd.DataFrame:
    """当日(JST)の 09:00–15:30 に限定。"""
    if df.empty:
        return df
    t = df["time"].dt.tz_convert(JP_TZ)
    today = pd.Timestamp.now(tz=JP_TZ).normalize()
    start = pd.to_datetime(f"{today.date()} {SESSION_START}", utc=False).tz_localize(JP_TZ)
    end   = pd.to_datetime(f"{today.date()} {SESSION_END}"  , utc=False).tz_localize(JP_TZ)
    m = (t >= start) & (t <= end)
    return df.loc[m].reset_index(drop=True)


def resample_to_minutes(df: pd.DataFrame, rule: str = "1min") -> pd.DataFrame:
    """time を index にして線形補間の簡易リサンプル。"""
    if df.empty:
        return df
    tmp = df.set_index("time").sort_index()
    # valueだけ対象（volumeは合計でも平均でも良いが、なければ0）
    out = tmp[["value"]].resample(rule).mean()
    out["value"] = out["value"].interpolate(limit_direction="both")
    out["volume"] = 0
    out = out.reset_index()
    return out


# ====== 描画ヘルパ ======
def _decorate_axes(ax, title: str, x_label: str, y_label: str):
    ax.set_title(title, color=TITLE, fontsize=20, pad=12)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=GRID_A)
    # スパイン
    for sp in ax.spines.values():
        sp.set_color(FG)


def save_png(fig: plt.Figure, out_path: str):
    # facecolor は rcParams で BG 指定済みだが、明示で統一
    fig.savefig(out_path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


# ====== 長期用（履歴） ======
def read_history(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["date", "value"])
    df = pd.read_csv(path)
    # 小文字化＆最小バリデーション
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "date" not in df.columns or "value" not in df.columns:
        return pd.DataFrame(columns=["date", "value"])
    df["date"] = pd.to_datetime(df["date"], utc=False, errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)
    return df


def plot_history_panel(ax, hist: pd.DataFrame, title: str):
    _decorate_axes(ax, title, "Date", "Index Value")
    hist = hist.sort_values("date")
    if len(hist) >= 2:
        ax.plot(hist["date"], hist["value"], linewidth=2.2, color=ACCENT)
    elif len(hist) == 1:
        ax.plot(hist["date"], hist["value"], marker="o", markersize=6, linewidth=0, color=ACCENT)
        y = hist["value"].iloc[0]
        ax.set_ylim(y - 0.1, y + 0.1)
        ax.text(0.5, 0.5, "Only 1 point (need ≥ 2)", transform=ax.transAxes,
                ha="center", va="center", alpha=0.5)
    else:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", alpha=0.5)


# ====== メイン ======
def main():
    index_key = os.environ.get("INDEX_KEY", "rbank9").strip().lower()
    index_name = index_key.upper().replace("_", "")

    outputs_dir = os.path.join("docs", "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    intraday_csv = os.path.join(outputs_dir, f"{index_key}_intraday.csv")
    history_csv  = os.path.join(outputs_dir, f"{index_key}_history.csv")

    # ---------- 1d ----------
    try:
        intraday = read_any_intraday(intraday_csv, JP_TZ, JP_TZ)
        intraday = clamp_to_session_today_jst(intraday)
        intraday = resample_to_minutes(intraday, "1min")
    except Exception as e:
        print(f"[WARN] intraday load failed: {e}")
        intraday = pd.DataFrame(columns=["time", "value", "volume"])

    fig, ax = plt.subplots(figsize=(16, 7), layout="constrained")
    _decorate_axes(ax, f"{index_name} (1d)", "Time", "Index Value")

    if not intraday.empty:
        ax.plot(intraday["time"], intraday["value"], linewidth=2.4, color=ACCENT)
    else:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", alpha=0.6)

    save_png(fig, os.path.join(outputs_dir, f"{index_key}_1d.png"))

    # ---------- 7d / 1m / 1y ----------
    history = read_history(history_csv)

    # 7d
    fig, ax = plt.subplots(figsize=(16, 7), layout="constrained")
    plot_history_panel(ax, history.tail(7), f"{index_name} (7d)")
    save_png(fig, os.path.join(outputs_dir, f"{index_key}_7d.png"))

    # 1m（暦30日で簡便に）
    fig, ax = plt.subplots(figsize=(16, 7), layout="constrained")
    plot_history_panel(ax, history.tail(30), f"{index_name} (1m)")
    save_png(fig, os.path.join(outputs_dir, f"{index_key}_1m.png"))

    # 1y（暦365日で簡便に）
    fig, ax = plt.subplots(figsize=(16, 7), layout="constrained")
    plot_history_panel(ax, history.tail(365), f"{index_name} (1y)")
    save_png(fig, os.path.join(outputs_dir, f"{index_key}_1y.png"))

    # 実行時刻メモ（CI の“出力有無確認”に便利）
    with open(os.path.join(outputs_dir, "_last_run.txt"), "w") as f:
        f.write(pd.Timestamp.now(tz=JP_TZ).isoformat())


if __name__ == "__main__":
    main()
