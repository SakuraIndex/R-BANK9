# scripts/long_charts.py
# -*- coding: utf-8 -*-
"""
Generate long-term charts (1d / 7d / 1m / 1y) for INDEX_KEY
with auto coloring (Up=GREEN / Down=RED).

Inputs
- docs/outputs/<index>_intraday.csv  (columns: time,value[,volume] OR wide-by-tickers)
- docs/outputs/<index>_history.csv   (columns: date,value)

Outputs
- docs/outputs/<index>_1d.png
- docs/outputs/<index>_7d.png
- docs/outputs/<index>_1m.png
- docs/outputs/<index>_1y.png
"""

from __future__ import annotations
import os
from typing import List, Optional

import pandas as pd
import pytz
import matplotlib
import matplotlib.pyplot as plt

# === Constants ===
JP_TZ = pytz.timezone("Asia/Tokyo")
SESSION_START = "09:00"
SESSION_END   = "15:30"

# Dark theme
BG = "#0E1117"
FG = "#E6E6E6"
TITLE  = "#f2b6c6"
GRID_A = 0.25

# Line colors (Up / Down)
GREEN = "#22c55e"   # 上昇
RED   = "#ef4444"   # 下落
NEUTRAL = "#94a3b8" # 同値 or データ不足

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

OUTPUTS_DIR = os.path.join("docs", "outputs")


# === Utilities ===
def _lower(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

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

def _auto_color(series: pd.Series) -> str:
    """Return GREEN if last>=first, RED if last<first, else NEUTRAL."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 2:
        return NEUTRAL
    first, last = s.iloc[0], s.iloc[-1]
    if pd.isna(first) or pd.isna(last):
        return NEUTRAL
    return GREEN if last >= first else RED

def read_any_intraday(path: str) -> pd.DataFrame:
    """
    Return columns: time (tz-aware JST), value, volume
    Accepts either long ("time,value[,volume]") or wide (tickers) formats.
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=["time", "value", "volume"])
    raw = pd.read_csv(path, dtype=str)
    if raw.empty:
        return pd.DataFrame(columns=["time", "value", "volume"])
    df = _lower(raw.copy())

    # remove comment-like columns starting with '#'
    drop = [c for c in df.columns if str(c).strip().startswith("#")]
    if drop:
        df = df.drop(columns=drop)

    tcol = _pick_time_col(df.columns.tolist())
    if tcol is None:
        raise KeyError(f"No time-like column in {path}")

    # guess value / volume
    vcol, volcol = None, None
    for c in df.columns:
        lc = c
        if lc in ("value", "index", "score") or ("value" in lc):
            vcol = c
        if lc == "volume" or ("volume" in lc):
            volcol = c

    # time to tz-aware JST
    t = pd.to_datetime(df[tcol], errors="coerce", utc=True)
    if t.dt.tz is None:  # naive -> JST
        t = pd.to_datetime(df[tcol], errors="coerce").dt.tz_localize(JP_TZ)
    else:
        t = t.dt.tz_convert(JP_TZ)

    out = pd.DataFrame({"time": t})

    if vcol is not None:
        out["value"] = pd.to_numeric(df[vcol], errors="coerce")
        out["volume"] = (
            pd.to_numeric(df[volcol], errors="coerce")
            if (volcol and volcol in df.columns) else 0
        )
    else:
        # wide → 等加重平均
        num_cols = []
        for c in df.columns:
            if c == tcol:
                continue
            as_num = pd.to_numeric(df[c], errors="coerce")
            if as_num.notna().sum() > 0:
                num_cols.append(c)
        if not num_cols:
            return pd.DataFrame(columns=["time", "value", "volume"])
        vals = df[num_cols].apply(lambda s: pd.to_numeric(s, errors="coerce"))
        out["value"] = vals.mean(axis=1)
        out["volume"] = 0

    return out.dropna(subset=["time", "value"]).sort_values("time").reset_index(drop=True)

def clamp_today_session_jst(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    today = pd.Timestamp.now(tz=JP_TZ).normalize()
    start = pd.Timestamp(f"{today.date()} {SESSION_START}", tz=JP_TZ)
    end   = pd.Timestamp(f"{today.date()} {SESSION_END}",   tz=JP_TZ)
    m = (df["time"] >= start) & (df["time"] <= end)
    return df.loc[m].reset_index(drop=True)

def resample_minutes(df: pd.DataFrame, rule: str = "1min") -> pd.DataFrame:
    if df.empty:
        return df
    tmp = df.set_index("time").sort_index()
    out = tmp[["value"]].resample(rule).mean()
    out["value"] = out["value"].interpolate(limit_direction="both")
    out["volume"] = 0
    return out.reset_index()

def read_history(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["date", "value"])
    df = pd.read_csv(path)
    df = _lower(df)
    if "date" not in df.columns or "value" not in df.columns:
        return pd.DataFrame(columns=["date", "value"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)

def _decorate(ax, title: str, xl: str, yl: str):
    ax.set_title(title, color=TITLE, fontsize=20, pad=12)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.grid(True, alpha=GRID_A)
    for sp in ax.spines.values():
        sp.set_color(FG)

def _save(fig, path: str):
    fig.savefig(path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


# === Main ===
def main():
    index_key = os.environ.get("INDEX_KEY", "rbank9").strip().lower()
    index_name = index_key.upper().replace("_", "")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    intraday_csv = os.path.join(OUTPUTS_DIR, f"{index_key}_intraday.csv")
    history_csv  = os.path.join(OUTPUTS_DIR, f"{index_key}_history.csv")

    # ---- 1d (intraday) ----
    try:
        i = read_any_intraday(intraday_csv)
        i = clamp_today_session_jst(i)
        i = resample_minutes(i, "1min")
    except Exception as e:
        print(f"[WARN] intraday load failed: {e}")
        i = pd.DataFrame(columns=["time", "value", "volume"])

    fig, ax = plt.subplots(figsize=(16, 7), layout="constrained")
    _decorate(ax, f"{index_name} (1d)", "Time", "Index Value")
    if not i.empty:
        color = _auto_color(i["value"])
        ax.plot(i["time"], i["value"], linewidth=2.4, color=color)
    else:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", alpha=0.6)
    _save(fig, os.path.join(OUTPUTS_DIR, f"{index_key}_1d.png"))

    # ---- 7d / 1m / 1y (daily history) ----
    h = read_history(history_csv)

    def plot_hist(tail_n: int, label: str, out: str):
        fig, ax = plt.subplots(figsize=(16, 7), layout="constrained")
        _decorate(ax, f"{index_name} ({label})", "Date", "Index Value")
        hh = h.tail(tail_n)
        if len(hh) >= 2:
            color = _auto_color(hh["value"])
            ax.plot(hh["date"], hh["value"], linewidth=2.2, color=color)
        elif len(hh) == 1:
            ax.plot(hh["date"], hh["value"], marker="o", markersize=6, linewidth=0, color=NEUTRAL)
            y = hh["value"].iloc[0]
            ax.set_ylim(y - 0.1, y + 0.1)
            ax.text(0.5, 0.5, "Only 1 point (need ≥ 2)", transform=ax.transAxes,
                    ha="center", va="center", alpha=0.5)
        else:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", alpha=0.5)
        _save(fig, os.path.join(OUTPUTS_DIR, out))

    plot_hist(7,   "7d", f"{index_key}_7d.png")
    plot_hist(30,  "1m", f"{index_key}_1m.png")
    plot_hist(365, "1y", f"{index_key}_1y.png")

    # 最終実行ログ
    with open(os.path.join(OUTPUTS_DIR, "_last_run.txt"), "w") as f:
        f.write(pd.Timestamp.now(tz=JP_TZ).isoformat())

if __name__ == "__main__":
    main()
