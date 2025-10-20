#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.dates import AutoDateLocator, AutoDateFormatter

INDEX_KEY = os.environ.get("INDEX_KEY", "rbank9")
OUT_DIR = Path("docs/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── ダークテーマ（白文字＋淡いグリッド） ─────────────────────────────
def apply_dark_theme(fig, ax):
    ax.set_facecolor("#111317")
    fig.patch.set_facecolor("#111317")
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 目盛・ラベルは白
    ax.tick_params(colors="#ffffff", labelsize=10)
    ax.xaxis.label.set_color("#ffffff")
    ax.yaxis.label.set_color("#ffffff")

    # グリッド（メジャー／マイナー）
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.18, color="#ffffff")
    ax.grid(True, which="minor", linestyle="-", linewidth=0.4, alpha=0.10, color="#ffffff")

# ── CSV ロード（先頭2列: ts, val 想定 / 1d, 7d, 1m, 1y で共通） ───────────────
def load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError(f"CSV must have >= 2 columns: {csv_path}")
    ts_col, val_col = df.columns[:2]
    df = df.rename(columns={ts_col: "ts", val_col: "val"})
    df["ts"] = pd.to_datetime(df["ts"], utc=False, errors="coerce")
    df = df.dropna(subset=["ts", "val"]).sort_values("ts").reset_index(drop=True)
    return df

# ── 1スパンを描画 ──────────────────────────────────────────────
def plot_one_span(df: pd.DataFrame, title: str, out_png: Path, ylabel: str):
    fig, ax = plt.subplots(figsize=(16, 8), dpi=110)
    apply_dark_theme(fig, ax)

    ax.set_title(title, fontsize=26, fontweight="bold", pad=18, color="#ffffff")
    ax.set_xlabel("Time", labelpad=10, color="#ffffff")
    ax.set_ylabel(ylabel, labelpad=10, color="#ffffff")

    ax.plot(df["ts"].values, df["val"].values, linewidth=2.6, color="#ff615a")

    # 時間軸（メジャー／マイナー）
    major = AutoDateLocator(minticks=5, maxticks=10)
    ax.xaxis.set_major_locator(major)
    ax.xaxis.set_major_formatter(AutoDateFormatter(major))
    ax.xaxis.set_minor_locator(MaxNLocator(nbins=50))

    # Y 軸は見やすい適当割
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.yaxis.set_minor_locator(MaxNLocator(nbins=7))

    fig.tight_layout()
    fig.savefig(out_png, facecolor=fig.get_facecolor())
    plt.close(fig)

# ── メイン ───────────────────────────────────────────────────
def main():
    spans = ["1d", "7d", "1m", "1y"]
    for span in spans:
        csv = OUT_DIR / f"{INDEX_KEY}_{span}.csv"
        if not csv.exists():
            continue
        df = load_csv(csv)

        # RBANK9 は level スケールで運用中（サイトの注記とも整合）
        # ラベルは画像タイトルにも合わせて "(1d level)" 等と表記
        out_png = OUT_DIR / f"{INDEX_KEY}_{span}.png"
        title  = f"{INDEX_KEY.upper()} ({span} level)"
        ylabel = "Index (level)"

        plot_one_span(df, title, out_png, ylabel)

if __name__ == "__main__":
    main()
