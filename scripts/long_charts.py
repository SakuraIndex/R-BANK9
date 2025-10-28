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

def theme(fig, ax):
    fig.set_size_inches(16, 8); fig.set_dpi(200)
    bg, fg, grid = "#111317", "#e7ebf3", "#ffffff"
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(colors=fg, labelsize=10)
    ax.yaxis.label.set_color(fg); ax.xaxis.label.set_color(fg); ax.title.set_color(fg)
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.18, color=grid)
    ax.grid(True, which="minor", linestyle="-", linewidth=0.4, alpha=0.10, color=grid)

def load_hist() -> pd.DataFrame:
    p = OUT_DIR / f"{INDEX_KEY}_history.csv"
    if not p.exists():
        return pd.DataFrame(columns=["ts","val"])
    df = pd.read_csv(p)
    cols = {c.lower(): c for c in df.columns}
    tcol = cols.get("ts", cols.get("date"))
    vcol = cols.get("val", cols.get("value"))
    if tcol is None or vcol is None:
        return pd.DataFrame(columns=["ts","val"])
    out = pd.DataFrame({
        "ts":  pd.to_datetime(df[tcol], errors="coerce", utc=False),
        "val": pd.to_numeric(df[vcol], errors="coerce")
    }).dropna().sort_values("ts")
    return out[~out["ts"].duplicated(keep="last")]

def span_window(df: pd.DataFrame, days: int) -> pd.DataFrame:
    if df.empty: return df
    last = df["ts"].max()
    return df[df["ts"] >= (last - pd.Timedelta(days=days))].copy()

def plot_level(d: pd.DataFrame, label: str, out_png: Path):
    fig, ax = plt.subplots(); theme(fig, ax)
    ax.set_title(f"{INDEX_KEY.upper()} ({label} level)", fontsize=26, fontweight="bold", pad=18)
    ax.set_xlabel("Time"); ax.set_ylabel("Level (index)")
    major = AutoDateLocator(minticks=5, maxticks=10)
    ax.xaxis.set_major_locator(major); ax.xaxis.set_major_formatter(AutoDateFormatter(major))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7)); ax.yaxis.set_minor_locator(MaxNLocator(nbins=50))
    n = len(d)
    if n == 0:
        ax.text(0.5, 0.5, "No history yet", ha="center", va="center", transform=ax.transAxes)
    elif n == 1:
        ax.scatter(d["ts"].values, d["val"].values, s=30)
        ax.text(0.02, 0.95, "Insufficient history (need ≥ 2 days)",
                ha="left", va="top", transform=ax.transAxes, fontsize=10, alpha=0.9)
        t = d["ts"].iloc[0]; ax.set_xlim(t - pd.Timedelta(days=3), t + pd.Timedelta(days=3))
    else:
        ax.plot(d["ts"].values, d["val"].values, linewidth=2.3)
    fig.tight_layout(); fig.savefig(out_png, facecolor=fig.get_facecolor(), bbox_inches="tight"); plt.close(fig)
    print(f"[long] wrote: {out_png}")

def main():
    df = load_hist()
    for label, days in [("7d",7), ("1m",30), ("1y",365)]:
        plot_level(span_window(df, days), label, OUT_DIR / f"{INDEX_KEY}_{label}.png")
    (OUT_DIR / "_last_run.txt").write_text(pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"))

if __name__ == "__main__":
    main()
