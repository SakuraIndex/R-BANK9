#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.dates import AutoDateLocator, AutoDateFormatter

INDEX_KEY = os.environ.get("INDEX_KEY", "rbank9")
OUT_DIR   = Path(os.environ.get("OUT_DIR", "docs/outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPS       = 0.2
CLAMP_PCT = 30.0

def apply_dark(fig, ax):
    fig.set_size_inches(16, 8)
    fig.set_dpi(200)
    fig.patch.set_facecolor("#111317")
    ax.set_facecolor("#111317")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(axis="both", colors="#ffffff", labelsize=10)
    ax.yaxis.label.set_color("#ffffff"); ax.xaxis.label.set_color("#ffffff"); ax.title.set_color("#ffffff")
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.18, color="#ffffff")
    ax.grid(True, which="minor", linestyle="-", linewidth=0.4, alpha=0.10, color="#ffffff")

def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists(): return pd.DataFrame()
    df = pd.read_csv(path)
    if df.shape[1] < 2: return pd.DataFrame()
    ts_col, val_col = df.columns[:2]
    df = df.rename(columns={ts_col:"ts", val_col:"val"})
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=False)
    df["val"] = pd.to_numeric(df["val"], errors="coerce")
    return df.dropna().sort_values("ts")

def pick_csv(span: str) -> Path:
    if span in ("7d","1m","1y"):
        p = OUT_DIR / f"{INDEX_KEY}_history.csv"
        if p.exists(): return p
    return OUT_DIR / f"{INDEX_KEY}_intraday.csv"

def clamp(p: float) -> float:
    if p >  CLAMP_PCT: return CLAMP_PCT
    if p < -CLAMP_PCT: return -CLAMP_PCT
    return p

def make_pct(df: pd.DataFrame, span: str) -> pd.DataFrame:
    if df.empty: return df
    if span == "1d":
        day = df["ts"].dt.floor("D").iloc[-1]
        d = df[df["ts"].dt.floor("D") == day].copy()
        if d.empty: return d
        # 始値（10:00以降の最初の値があればそれ、なければ当日最初）
        base = d.loc[(d["ts"].dt.hour > 10) | ((d["ts"].dt.hour == 10) & (d["ts"].dt.minute >= 0)), "val"]
        base = float(base.iloc[0]) if not base.empty else float(d.iloc[0]["val"])
        denom = max(abs(base), EPS)
        d["pct"] = (d["val"] - base) / denom * 100.0
        d["pct"] = d["pct"].clip(lower=-CLAMP_PCT, upper=CLAMP_PCT)
        return d

    win_days = {"7d":7,"1m":30,"1y":365}[span]
    last = df["ts"].max()
    w = df[df["ts"] >= (last - pd.Timedelta(days=win_days))].copy()
    if w.empty: return w
    base = float(w.iloc[0]["val"])
    denom = max(abs(base), EPS)       # ← 基準日の値のみを分母に
    w["pct"] = (w["val"] - base) / denom * 100.0
    w["pct"] = w["pct"].clip(lower=-CLAMP_PCT, upper=CLAMP_PCT)
    return w

def plot_one(dfp: pd.DataFrame, title: str, out_png: Path):
    fig, ax = plt.subplots()
    apply_dark(fig, ax)
    ax.set_title(title, fontsize=26, fontweight="bold", pad=18)
    ax.set_xlabel("Time", labelpad=10); ax.set_ylabel("Change (%)", labelpad=10)
    ax.plot(dfp["ts"].values, dfp["pct"].values, linewidth=2.6)
    major = AutoDateLocator(minticks=5, maxticks=10)
    ax.xaxis.set_major_locator(major); ax.xaxis.set_major_formatter(AutoDateFormatter(major))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7)); ax.yaxis.set_minor_locator(MaxNLocator(nbins=50))
    fig.tight_layout()
    fig.savefig(out_png, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"[long] wrote {out_png}")

def main():
    for span in ["1d","7d","1m","1y"]:
        csv = pick_csv(span)
        df  = load_csv(csv)
        if df.empty: 
            print(f"[long] skip {span}: no data ({csv})")
            continue
        dfp = make_pct(df, span)
        if dfp.empty or "pct" not in dfp:
            print(f"[long] skip {span}: no pct")
            continue
        plot_one(dfp, f"{INDEX_KEY.upper()} ({span})", OUT_DIR / f"{INDEX_KEY}_{span}.png")

if __name__ == "__main__":
    main()
