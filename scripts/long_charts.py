#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.dates import AutoDateLocator, AutoDateFormatter

INDEX_KEY = os.environ.get("INDEX_KEY", "rbank9")
OUT_DIR = Path(os.environ.get("OUT_DIR", "docs/outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPS = 0.2  # 極端な 0 除算回避用の閾値

# ── ダークテーマ（薄いグリッド＆白い軸文字・目盛） ─────────────────────────
def apply_dark_theme(fig, ax):
    # 背景
    fig.patch.set_facecolor("#111317")
    ax.set_facecolor("#111317")

    # スパインは消す
    for sp in ax.spines.values():
        sp.set_visible(False)

    # 目盛とラベルを白系に
    ax.tick_params(colors="#cfd3dc", labelsize=10)
    ax.yaxis.label.set_color("#e7ebf3")
    ax.xaxis.label.set_color("#e7ebf3")

    # メジャーグリッド（薄い線）
    ax.grid(
        True,
        which="major",
        linestyle="-",
        linewidth=0.6,
        alpha=0.18,
        color="#ffffff",
    )
    # マイナーグリッド（さらに薄く）
    ax.grid(
        True,
        which="minor",
        linestyle="-",
        linewidth=0.4,
        alpha=0.10,
        color="#ffffff",
    )

# ── CSV 読み込み（先頭2列 = ts, val を想定） ──────────────────────────────
def load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError(f"CSV needs >=2 columns: {csv_path}")
    ts_col, val_col = df.columns[:2]
    df = df.rename(columns={ts_col: "ts", val_col: "val"})
    df["ts"] = pd.to_datetime(df["ts"], utc=False, errors="coerce")
    df = df.dropna(subset=["ts", "val"]).sort_values("ts").reset_index(drop=True)
    return df

# ── 1日のベース（10:00以降で |val|>=EPS の最初、なければ当日初値。ただし分母ゼロを回避） ─
def stable_baseline(df_day: pd.DataFrame) -> float | None:
    if df_day.empty:
        return None
    mask = (df_day["ts"].dt.hour > 10) | ((df_day["ts"].dt.hour == 10) & (df_day["ts"].dt.minute >= 0))
    cand = df_day.loc[mask & (df_day["val"].abs() >= EPS)]
    if not cand.empty:
        return float(cand.iloc[0]["val"])
    # fallback: 最初の |val|>=EPS
    cand2 = df_day.loc[df_day["val"].abs() >= EPS]
    if not cand2.empty:
        return float(cand2.iloc[0]["val"])
    return None

def calc_sane_pct(base: float, close: float) -> float:
    # 分母ゼロ防止
    denom = max(abs(base), EPS)
    return (close - base) / denom * 100.0

# ── span別に % 系列を作る（"1d","7d","1m","1y"） ────────────────────────────
def make_pct_series(df: pd.DataFrame, span: str) -> pd.DataFrame:
    if df.empty:
        return df

    if span == "1d":
        the_day = df["ts"].dt.floor("D").iloc[-1]
        df_day = df[df["ts"].dt.floor("D") == the_day].copy()
        if df_day.empty:
            return df_day
        base = stable_baseline(df_day)
        if base is None:
            return pd.DataFrame()
        df_day["pct"] = df_day["val"].apply(lambda x: calc_sane_pct(base, x))
        return df_day

    last = df["ts"].max()
    days_map = {"7d": 7, "1m": 30, "1y": 365}
    days = days_map.get(span, 7)
    df_span = df[df["ts"] >= (last - pd.Timedelta(days=days))].copy()
    if df_span.empty:
        return df_span
    base = float(df_span.iloc[0]["val"])
    df_span["pct"] = df_span["val"].apply(lambda x: calc_sane_pct(base, x))
    return df_span

# ── プロット ─────────────────────────────────────────────────────────────
def plot_one_span(df_pct: pd.DataFrame, title: str, out_png: Path):
    fig, ax = plt.subplots(figsize=(16, 8), dpi=110)
    apply_dark_theme(fig, ax)

    ax.set_title(title, fontsize=26, fontweight="bold", pad=18)
    ax.set_xlabel("Time", labelpad=10)
    ax.set_ylabel("Change (%)", labelpad=10)

    ax.plot(df_pct["ts"].values, df_pct["pct"].values, linewidth=2.6, color="#ff615a")

    major = AutoDateLocator(minticks=5, maxticks=10)
    ax.xaxis.set_major_locator(major)
    ax.xaxis.set_major_formatter(AutoDateFormatter(major))
    ax.xaxis.set_minor_locator(MaxNLocator(nbins=7))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))

    fig.tight_layout()
    # ← これがないと背景が黒で保存されずグレーになる環境があります
    fig.savefig(out_png, facecolor=fig.get_facecolor())
    plt.close(fig)

def main():
    spans = ["1d", "7d", "1m", "1y"]
    csv = OUT_DIR / f"{INDEX_KEY}_1d.csv"
    if not csv.exists():
        print(f"CSV not found: {csv}")
        return

    df = load_csv(csv)
    for span in spans:
        df_pct = make_pct_series(df, span)
        if df_pct.empty or "pct" not in df_pct:
            continue
        out_png = OUT_DIR / f"{INDEX_KEY}_{span}.png"
        title = f"{INDEX_KEY.upper()} ({span})"
        plot_one_span(df_pct, title, out_png)

if __name__ == "__main__":
    main()
