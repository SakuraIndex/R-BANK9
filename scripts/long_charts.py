# scripts/long_charts.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re
import pandas as pd
import pytz
import matplotlib
import matplotlib.pyplot as plt

JP = pytz.timezone("Asia/Tokyo")
BG, FG, TITLE = "#0E1117", "#E6E6E6", "#f2b6c6"
UP, DOWN, GRID_A = "#3bd6c6", "#ff6b6b", 0.25
matplotlib.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": FG,
    "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
    "text.color": FG, "grid.color": FG, "savefig.facecolor": BG,
})

OUTDIR = os.path.join("docs", "outputs")
REPO = os.getenv("GITHUB_REPOSITORY", "")
KEY = REPO.split("/")[-1].lower().replace("-", "_") if REPO else "index"
NAME = KEY.upper()

def _lower(df): df.columns=[str(c).strip() for c in df.columns]; return df
def _find_time_col(cols):
    for c in cols:
        if re.search(r"time|日時|date|datetime|timestamp|時刻", str(c), re.I): return c
    return cols[0] if cols else None

def read_intraday(path:str)->pd.DataFrame:
    if not os.path.exists(path): 
        return pd.DataFrame(columns=["time","value"])
    raw = pd.read_csv(path, dtype=str)
    if raw.empty:
        return pd.DataFrame(columns=["time","value"])
    df = _lower(raw)
    tcol = _find_time_col(df.columns)
    if tcol is None: return pd.DataFrame(columns=["time","value"])
    # 候補列
    candidates = [c for c in df.columns if re.search(rf"{KEY}|value", c, re.I)]
    vcol = candidates[0] if candidates else df.columns[-1]
    t = pd.to_datetime(df[tcol], errors="coerce", utc=True)
    if t.dt.tz is None:
        t = t.dt.tz_localize("UTC")
    t = t.dt.tz_convert(JP)
    out = pd.DataFrame({"time": t, "value": pd.to_numeric(df[vcol], errors="coerce")})
    return out.dropna().sort_values("time")

def calc_delta(series: pd.Series) -> float | None:
    """％変化率：(終値÷始値−1)*100"""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 2: return None
    base, last = float(s.iloc[0]), float(s.iloc[-1])
    if base == 0: return None
    return (last / base - 1.0) * 100.0

def pick_window(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    now = pd.Timestamp.now(tz=JP)
    beg = now - pd.Timedelta(hours=24)
    w = df[(df["time"]>=beg)&(df["time"]<=now)]
    return w if not w.empty else df.tail(600)

def decorate(ax, title, xl, yl):
    ax.set_title(title, color=TITLE, fontsize=20, pad=12)
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(True, alpha=GRID_A)
    for sp in ax.spines.values(): sp.set_color(FG)

def save(fig, path):
    fig.savefig(path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    csv = os.path.join(OUTDIR, f"{KEY}_intraday.csv")
    hist = os.path.join(OUTDIR, f"{KEY}_history.csv")

    df = read_intraday(csv)
    df = pick_window(df)
    if not df.empty:
        df = (df.set_index("time").resample("1min").mean()
              .interpolate(limit_direction="both").reset_index())
    delta = calc_delta(df["value"]) if not df.empty else None
    color = UP if (delta is not None and delta >= 0) else DOWN

    # 1d chart
    fig, ax = plt.subplots(figsize=(16,7), layout="constrained")
    decorate(ax, f"{NAME} (1d)", "Time", "Index Value")
    if not df.empty:
        ax.plot(df["time"], df["value"], lw=2.0, color=color)
    else:
        ax.text(0.5,0.5,"No data", ha="center", va="center", transform=ax.transAxes)
    save(fig, os.path.join(OUTDIR, f"{KEY}_1d.png"))

    # 出力テキスト
    txt = f"{NAME} 1d: {(0.0 if delta is None else delta):+0.2f}%"
    with open(os.path.join(OUTDIR, f"{KEY}_post_intraday.txt"), "w", encoding="utf-8") as f:
        f.write(txt)

if __name__ == "__main__":
    main()
