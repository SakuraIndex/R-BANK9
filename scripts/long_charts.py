#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate long-term charts (1d / 7d / 1m / 1y)

- 1d  : intraday CSV を読み、直近セッション(JST) 09:00–15:30だけを描画
- 7d/1m/1y: history CSV (date,value) を描画
- 互換のため 1d を <index>_intraday.png としても保存

出力:
  docs/outputs/<index>_1d.png
  docs/outputs/<index>_7d.png
  docs/outputs/<index>_1m.png
  docs/outputs/<index>_1y.png
  docs/outputs/<index>_intraday.png  (1dと同じ)
"""

from __future__ import annotations
import os
import shutil
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
ACCENT = "#3bd6c6"
TITLE  = "#f2b6c6"
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
    cols_lower = [str(c).strip().lower() for c in df.columns]
    df.columns = cols_lower
    return cols_lower

def _pick_time_col(cols_lower: List[str]) -> Optional[str]:
    for k in ("time", "timestamp", "date", "datetime"):
        if k in cols_lower:
            return k
    for c in cols_lower:
        if c.startswith("unnamed") and (": 0" in c or c.endswith("0")):
            return c
    for c in cols_lower:
        if ("time" in c) or ("date" in c):
            return c
    return None

def parse_time_any(x, raw_tz: str, display_tz: str):
    """
    文字列/数値/epochを受け、naiveは raw_tz とみなしてローカライズ→display_tzに変換。
    """
    ts = pd.to_datetime(x, errors="coerce")  # utc=True は使わない
    if pd.isna(ts):
        return pd.NaT
    if ts.tzinfo is None:
        ts = ts.tz_localize(raw_tz)
    return ts.tz_convert(display_tz)

def read_any_intraday(path: Optional[str], raw_tz: str, display_tz: str) -> pd.DataFrame:
    """
    intraday CSV を読み、「time」「value」「volume」を返す。
    - time/value/volume 形式 or 銘柄横並び形式（等加重平均）を自動解釈
    - 先頭 "Unnamed: 0" も time として扱う
    """
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=["time", "value", "volume"])

    df = pd.read_csv(path, dtype=str)
    # コメント列（先頭が #）は除去
    drop_cols = [c for c in df.columns if str(c).strip().startswith("#")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    cols_lower = _lower_cols(df)
    tcol = _pick_time_col(cols_lower)
    if tcol is None:
        raise KeyError(f"No time-like column found. columns={list(df.columns)}")

    # value/volume がある場合
    vcol = None
    volcol = None
    for c in df.columns:
        lc = str(c).lower()
        if lc in ("value", "index") or ("value" in lc) or ("close" in lc):
            vcol = c
        if lc == "volume" or ("volume" in lc):
            volcol = c

    out = pd.DataFrame()
    out["time"] = df[tcol].apply(lambda x: parse_time_any(x, raw_tz, display_tz))

    if vcol is not None:
        out["value"] = pd.to_numeric(df[vcol], errors="coerce")
        out["volume"] = pd.to_numeric(df[volcol], errors="coerce") if (volcol and volcol in df.columns) else 0
    else:
        # 銘柄横並び: time 以外を数値化 → 等加重平均
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

def clamp_to_latest_session_jst(df: pd.DataFrame) -> pd.DataFrame:
    """
    データ中の「最後のタイムスタンプの“日付(JST)”」をその日のセッションとして抽出。
    当日データが無い日・休日でも、直近営業日のセッションが出せる。
    """
    if df.empty:
        return df
    t_jst = df["time"].dt.tz_convert(JP_TZ)
    last_day = t_jst.max().date()
    start = pd.Timestamp(f"{last_day} {SESSION_START}", tz=JP_TZ)
    end   = pd.Timestamp(f"{last_day} {SESSION_END}",   tz=JP_TZ)
    m = (t_jst >= start) & (t_jst <= end)
    sliced = df.loc[m].reset_index(drop=True)
    if sliced.empty:
        # 念のため、同日のデータだけ確保（時間帯外でもプロットはできる）
        sliced = df.loc[t_jst.dt.date == last_day].reset_index(drop=True)
    return sliced

def resample_to_minutes(df: pd.DataFrame, rule: str = "1min") -> pd.DataFrame:
    if df.empty:
        return df
    tmp = df.set_index("time").sort_index()
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
    for sp in ax.spines.values():
        sp.set_color(FG)

def save_png(fig: plt.Figure, out_path: str):
    fig.savefig(out_path, facecolor=BG, bbox_inches="tight", dpi=150)
    plt.close(fig)

# ====== 長期用（履歴） ======
def read_history(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["date", "value"])
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "date" not in df.columns or "value" not in df.columns:
        return pd.DataFrame(columns=["date", "value"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
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
        y = float(hist["value"].iloc[0])
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
        print(f"[INFO] intraday rows: {len(intraday)} "
              f"min={intraday['time'].min() if not intraday.empty else None} "
              f"max={intraday['time'].max() if not intraday.empty else None}")
        intraday = clamp_to_latest_session_jst(intraday)
        intraday = resample_to_minutes(intraday, "1min")
        print(f"[INFO] intraday after clamp/resample rows: {len(intraday)}")
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
    out_1d = os.path.join(outputs_dir, f"{index_key}_1d.png")
    save_png(fig, out_1d)

    # 互換：intraday.png も出す
    try:
        shutil.copyfile(out_1d, os.path.join(outputs_dir, f"{index_key}_intraday.png"))
    except Exception as e:
        print(f"[WARN] copy intraday.png failed: {e}")

    # ---------- 7d / 1m / 1y ----------
    history = read_history(history_csv)
    print(f"[INFO] history rows: {len(history)} "
          f"min={history['date'].min() if not history.empty else None} "
          f"max={history['date'].max() if not history.empty else None}")

    # 7d（直近7営業日分）
    fig, ax = plt.subplots(figsize=(16, 7), layout="constrained")
    plot_history_panel(ax, history.tail(7), f"{index_name} (7d)")
    save_png(fig, os.path.join(outputs_dir, f"{index_key}_7d.png"))

    # 1m（直近30日）
    fig, ax = plt.subplots(figsize=(16, 7), layout="constrained")
    plot_history_panel(ax, history.tail(30), f"{index_name} (1m)")
    save_png(fig, os.path.join(outputs_dir, f"{index_key}_1m.png"))

    # 1y（直近365日）
    fig, ax = plt.subplots(figsize=(16, 7), layout="constrained")
    plot_history_panel(ax, history.tail(365), f"{index_name} (1y)")
    save_png(fig, os.path.join(outputs_dir, f"{index_key}_1y.png"))

    with open(os.path.join(outputs_dir, "_last_run.txt"), "w") as f:
        f.write(pd.Timestamp.now(tz=JP_TZ).isoformat())

if __name__ == "__main__":
    main()
