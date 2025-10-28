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

# ────────── theme ──────────
def apply_dark(fig, ax):
    fig.set_size_inches(16, 8)
    fig.set_dpi(200)
    bg = "#111317"
    fg = "#e7ebf3"
    grid = "#ffffff"
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(axis="both", colors=fg, labelsize=10)
    ax.yaxis.label.set_color(fg)
    ax.xaxis.label.set_color(fg)
    ax.title.set_color(fg)
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.18, color=grid)
    ax.grid(True, which="minor", linestyle="-", linewidth=0.4, alpha=0.10, color=grid)

def load_history() -> pd.DataFrame:
    csv = OUT_DIR / f"{INDEX_KEY}_history.csv"
    if not csv.exists():
        return pd.DataFrame(columns=["ts", "val"])
    df = pd.read_csv(csv)
    # 許容する典型フォーマット: date,value または ts,val
    cols = {c.lower(): c for c in df.columns}
    tcol = cols.get("ts", cols.get("date"))
    vcol = cols.get("val", cols.get("value"))
    if tcol is None or vcol is None:
        return pd.DataFrame(columns=["ts", "val"])
    out = pd.DataFrame({
        "ts":  pd.to_datetime(df[tcol], errors="coerce", utc=False),
        "val": pd.to_numeric(df[vcol], errors="coerce")
    }).dropna().sort_values("ts").reset_index(drop=True)
    # 重複 ts は後勝ち
    out = out[~out["ts"].duplicated(keep="last")]
    return out

def window(df: pd.DataFrame, days: int) -> pd.DataFrame:
    if df.empty:
        return df
    last = df["ts"].max()
    return df[df["ts"] >= (last - pd.Timedelta(days=days))].copy()

def plot_level(df: pd.DataFrame, title: str, ylabel: str, out_png: Path):
    fig, ax = plt.subplots()
    apply_dark(fig, ax)
    ax.set_title(title, fontsize=26, fontweight="bold", pad=18)
    ax.set_xlabel("Time", labelpad=10)
    ax.set_ylabel(ylabel, labelpad=10)

    # X 軸フォーマット
    major = AutoDateLocator(minticks=5, maxticks=10)
    ax.xaxis.set_major_locator(major)
    ax.xaxis.set_major_formatter(AutoDateFormatter(major))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.yaxis.set_minor_locator(MaxNLocator(nbins=50))

    n = len(df)
    if n == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    elif n == 1:
        # 1 点しか無い時は点だけ表示（水平線で誤解させない）
        ax.scatter(df["ts"].values, df["val"].values, s=30)
        ax.text(0.02, 0.95, "Insufficient history (need ≥ 2 days)",
                ha="left", va="top", transform=ax.transAxes, fontsize=10, alpha=0.9)
        # 1 点が中央に来るように軽く余白
        t = df["ts"].iloc[0]
        ax.set_xlim(t - pd.Timedelta(days=3), t + pd.Timedelta(days=3))
    else:
        # 2 点以上: 通常のライン
        ax.plot(df["ts"].values, df["val"].values, linewidth=2.3)

    fig.tight_layout()
    fig.savefig(out_png, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"[long] wrote: {out_png}")

def main():
    df = load_history()
    # 7d / 1m / 1y は「レベル」を描画
    spans = [("7d", 7), ("1m", 30), ("1y", 365)]
    for label, days in spans:
        d = window(df, days)
        out_png = OUT_DIR / f"{INDEX_KEY}_{label}.png"
        plot_level(d, f"{INDEX_KEY.upper()} ({label} level)", "Level (index)", out_png)

    # _last_run.txt もここで更新（workflow 側でも可）
    (OUT_DIR / "_last_run.txt").write_text(pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"))

if __name__ == "__main__":
    main()
