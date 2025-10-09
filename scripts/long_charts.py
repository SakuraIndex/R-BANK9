#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Long-term charts generator with auto-append to history
  - 1d: 当日(JST) 09:00–15:30 のみ表示（分足リサンプリング）
  - 7d/1m/1y: docs/outputs/<index>_history.csv（date,value）を使用
  - 実行時に intraday の最新値で本日の履歴を追記/更新（重複日付は上書き）
出力:
  docs/outputs/<index>_1d.png
  docs/outputs/<index>_7d.png
  docs/outputs/<index>_1m.png
  docs/outputs/<index>_1y.png
"""

from __future__ import annotations
import os
from typing import Optional, List

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# ====== 定数 ======
JP_TZ = "Asia/Tokyo"
SESSION_START = "09:00"
SESSION_END   = "15:30"

# ダークテーマ色
BG = "#0E1117"
FG = "#E6E6E6"
ACCENT = "#3bd6c6"   # ライン
TITLE  = "#f2b6c6"   # タイトル
GRID_A = 0.25

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
    cols = [str(c).strip().lower() for c in df.columns]
    df.columns = cols
    return cols

def _pick_time_col(cols: List[str]) -> Optional[str]:
    for k in ("time", "timestamp", "date", "datetime"):
        if k in cols:
            return k
    for c in cols:
        if c.startswith("unnamed") and ": 0" in c:
            return c
    for c in cols:
        if ("time" in c) or ("date" in c):
            return c
    return None

def read_any_intraday(path: Optional[str]) -> pd.DataFrame:
    """intraday CSV（単一列 or 板形式）を読み込み value を決定"""
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=["time", "value", "volume"])

    df = pd.read_csv(path, dtype=str)
    # 先頭が # の列はコメント扱いで除去
    drop_cols = [c for c in df.columns if str(c).strip().startswith("#")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    cols = _lower_cols(df)
    tcol = _pick_time_col(cols)
    if tcol is None:
        raise KeyError(f"No time-like column. columns={list(df.columns)}")

    # time
    t = pd.to_datetime(df[tcol], utc=True, errors="coerce")
    # もし tz が無い/混在でも UTC に寄せてから JST に変換
    t = t.dt.tz_convert(JP_TZ)
    out = pd.DataFrame({"time": t})

    # 単一 value/volume があるか？
    vcol = None
    volcol = None
    for c in df.columns:
        lc = str(c).lower()
        if lc in ("value", "index") or "value" in lc:
            vcol = c
        if lc == "volume" or ("volume" in lc):
            volcol = c

    if vcol is not None:
        out["value"]  = pd.to_numeric(df[vcol], errors="coerce")
        out["volume"] = pd.to_numeric(df[volcol], errors="coerce") if (volcol and volcol in df.columns) else 0
    else:
        # 板形式：time 以外の数値列の平均
        num_cols = []
        for c in df.columns:
            if c == tcol:
                continue
            series = pd.to_numeric(df[c], errors="coerce")
            if series.notna().sum() > 0:
                num_cols.append(c)
        if not num_cols:
            return pd.DataFrame(columns=["time", "value", "volume"])
        vals = df[num_cols].apply(lambda s: pd.to_numeric(s, errors="coerce"))
        out["value"]  = vals.mean(axis=1)
        out["volume"] = 0

    out = out.dropna(subset=["time", "value"]).sort_values("time").reset_index(drop=True)
    return out

def clamp_to_today_session_jst(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    t = df["time"].dt.tz_convert(JP_TZ)
    today = pd.Timestamp.now(tz=JP_TZ).normalize()
    start = pd.to_datetime(f"{today.date()} {SESSION_START}", utc=False).tz_localize(JP_TZ)
    end   = pd.to_datetime(f"{today.date()} {SESSION_END}"  , utc=False).tz_localize(JP_TZ)
    m = (t >= start) & (t <= end)
    return df.loc[m].reset_index(drop=True)

def resample_minutes(df: pd.DataFrame, rule: str = "1min") -> pd.DataFrame:
    if df.empty:
        return df
    tmp = df.set_index("time").sort_index()
    out = tmp[["value"]].resample(rule).mean()
    out["value"] = out["value"].interpolate(limit_direction="both")
    out["volume"] = 0
    return out.reset_index()

def _decorate(ax, title: str, xlab: str, ylab: str):
    ax.set_title(title, color=TITLE, fontsize=20, pad=12)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.grid(True, alpha=GRID_A)
    for sp in ax.spines.values():
        sp.set_color(FG)

def save(fig: plt.Figure, path: str):
    fig.savefig(path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)

# ====== 履歴CSV ======
def read_history(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["date", "value"])
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "date" not in df.columns or "value" not in df.columns:
        return pd.DataFrame(columns=["date", "value"])
    df["date"]  = pd.to_datetime(df["date"], utc=False, errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)
    return df

def append_or_update_today(history_csv: str, intraday_today: pd.DataFrame, keep_last: int = 1000):
    """当日の最終値で履歴を追記/更新（重複日付は上書き）。"""
    if intraday_today.empty:
        return
    today_date = pd.Timestamp.now(tz=JP_TZ).normalize().date()
    last_val = float(intraday_today["value"].tail(1).iloc[0])

    hist = read_history(history_csv)
    if hist.empty:
        hist = pd.DataFrame({"date": [pd.Timestamp(today_date)], "value": [last_val]})
    else:
        hist["date"] = pd.to_datetime(hist["date"]).dt.normalize()
        mask = hist["date"] == pd.Timestamp(today_date)
        if mask.any():
            hist.loc[mask, "value"] = last_val  # 上書き
        else:
            hist = pd.concat([hist, pd.DataFrame({"date": [pd.Timestamp(today_date)], "value": [last_val]})],
                             ignore_index=True)
    # 整理
    hist = hist.dropna(subset=["date", "value"]).drop_duplicates(subset=["date"], keep="last")
    hist = hist.sort_values("date").tail(keep_last).reset_index(drop=True)
    # 保存（dateは日付のみの文字列に）
    hist_out = hist.copy()
    hist_out["date"] = hist_out["date"].dt.strftime("%Y-%m-%d")
    os.makedirs(os.path.dirname(history_csv), exist_ok=True)
    hist_out.to_csv(history_csv, index=False)

def slice_last_days(hist: pd.DataFrame, days: int) -> pd.DataFrame:
    if hist.empty:
        return hist
    end = pd.Timestamp.now(tz=JP_TZ).normalize()
    start = end - pd.Timedelta(days=days-1)
    return hist[(hist["date"] >= start) & (hist["date"] <= end)].copy()

def plot_hist(ax, hist: pd.DataFrame, title: str):
    _decorate(ax, title, "Date", "Index Value")
    if len(hist) >= 2:
        ax.plot(hist["date"], hist["value"], linewidth=2.2, color=ACCENT)
    elif len(hist) == 1:
        ax.plot(hist["date"], hist["value"], marker="o", markersize=6, linewidth=0, color=ACCENT)
        y = float(hist["value"].iloc[0])
        ax.set_ylim(y - 0.1, y + 0.1)
        ax.text(0.5, 0.5, "Only 1 point (need ≥ 2)", transform=ax.transAxes,
                ha="center", va="center", alpha=0.55)
    else:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", alpha=0.55)

# ====== メイン ======
def main():
    index_key = os.environ.get("INDEX_KEY", "rbank9").strip().lower()
    index_name = index_key.upper().replace("_", "")

    out_dir = os.path.join("docs", "outputs")
    os.makedirs(out_dir, exist_ok=True)

    intraday_csv = os.path.join(out_dir, f"{index_key}_intraday.csv")
    history_csv  = os.path.join(out_dir, f"{index_key}_history.csv")

    # --- intraday 読込 → 当日セッション抽出 → 分足化
    try:
        intraday = read_any_intraday(intraday_csv)
        intraday_today = clamp_to_today_session_jst(intraday)
        intraday_today = resample_minutes(intraday_today, "1min")
    except Exception as e:
        print(f"[WARN] intraday load failed: {e}")
        intraday_today = pd.DataFrame(columns=["time", "value", "volume"])

    # --- 履歴CSVに 今日の最終値 を追記/更新
    try:
        append_or_update_today(history_csv, intraday_today)
    except Exception as e:
        print(f"[WARN] history append failed: {e}")

    # --- 1d 描画
    fig, ax = plt.subplots(figsize=(16, 7), layout="constrained")
    _decorate(ax, f"{index_name} (1d)", "Time", "Index Value")
    if not intraday_today.empty:
        ax.plot(intraday_today["time"], intraday_today["value"], linewidth=2.4, color=ACCENT)
    else:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", alpha=0.6)
    save(fig, os.path.join(out_dir, f"{index_key}_1d.png"))

    # --- 7d / 1m / 1y 描画（追記後の履歴で）
    hist = read_history(history_csv)

    # 7d（直近7日）
    fig, ax = plt.subplots(figsize=(16, 7), layout="constrained")
    plot_hist(ax, slice_last_days(hist, 7), f"{index_name} (7d)")
    save(fig, os.path.join(out_dir, f"{index_key}_7d.png"))

    # 1m（直近30日）
    fig, ax = plt.subplots(figsize=(16, 7), layout="constrained")
    plot_hist(ax, slice_last_days(hist, 30), f"{index_name} (1m)")
    save(fig, os.path.join(out_dir, f"{index_key}_1m.png"))

    # 1y（直近365日）
    fig, ax = plt.subplots(figsize=(16, 7), layout="constrained")
    plot_hist(ax, slice_last_days(hist, 365), f"{index_name} (1y)")
    save(fig, os.path.join(out_dir, f"{index_key}_1y.png"))

    with open(os.path.join(out_dir, "_last_run.txt"), "w") as f:
        f.write(pd.Timestamp.now(tz=JP_TZ).isoformat())

if __name__ == "__main__":
    main()
