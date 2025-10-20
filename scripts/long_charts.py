#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.dates import AutoDateLocator, AutoDateFormatter

# ====== 環境変数 / 出力先 ======
INDEX_KEY = os.environ.get("INDEX_KEY", "rbank9")
OUT_DIR = Path("docs/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ====== ダークテーマ（グリッド強化＋軸文字白） ======
def apply_dark_theme(fig, ax):
    # 背景
    fig.patch.set_facecolor("#111317")
    ax.set_facecolor("#111317")

    # 枠線は消す
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 目盛とラベル（白系で統一）
    ax.tick_params(colors="#cfd3dc", labelsize=10, length=4)
    ax.xaxis.label.set_color("#cfd3dc")
    ax.yaxis.label.set_color("#cfd3dc")
    ax.title.set_color("#ffffff")

    # グリッド（メジャー/マイナー両方）
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.18, color="#ffffff")
    ax.grid(True, which="minor", linestyle="-", linewidth=0.4, alpha=0.10, color="#ffffff")

# ====== CSV 読み込み（ts, val の2列を想定） ======
def load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError(f"CSV must have >= 2 columns: {csv_path}")
    ts_col, val_col = df.columns[:2]
    df = df.rename(columns={ts_col: "ts", val_col: "val"})
    df["ts"] = pd.to_datetime(df["ts"], utc=False, errors="coerce")
    df = df.dropna(subset=["ts", "val"]).sort_values("ts").reset_index(drop=True)
    return df

# ====== 描画（1スパン） ======
def plot_one_span(df: pd.DataFrame, title: str, out_png: Path):
    fig, ax = plt.subplots(figsize=(16, 8), dpi=110)
    apply_dark_theme(fig, ax)

    ax.set_title(title, fontsize=26, fontweight="bold", pad=18)
    ax.set_xlabel("Time", labelpad=10)
    ax.set_ylabel("Index (level)", labelpad=10)

    ax.plot(df["ts"].values, df["val"].values, linewidth=2.6, color="#ff615a")

    # 軸の日付ロケータ/フォーマッタ
    major = AutoDateLocator(minticks=5, maxticks=10)
    ax.xaxis.set_major_locator(major)
    ax.xaxis.set_major_formatter(AutoDateFormatter(major))

    # マイナー目盛（横方向の薄い線を出すため）
    ax.xaxis.set_minor_locator(MaxNLocator(nbins=50))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7))

    # Y軸もマイナー目盛を有効にして薄い横線が出るように
    ax.yaxis.set_minor_locator(MaxNLocator(nbins=50))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))

    fig.tight_layout()
    fig.savefig(out_png, facecolor=fig.get_facecolor())
    plt.close(fig)

def main():
    # 1d（レベル）のみ描画（他スパンがある場合は必要に応じて増やしてください）
    csv = OUT_DIR / f"{INDEX_KEY}_1d.csv"
    if not csv.exists():
        print(f"[WARN] not found: {csv}")
        return

    df = load_csv(csv)
    if df.empty:
        print(f"[WARN] empty csv: {csv}")
        return

    out_png = OUT_DIR / f"{INDEX_KEY}_1d.png"
    title = f"{INDEX_KEY.upper()} (1d level)"
    plot_one_span(df, title, out_png)

if __name__ == "__main__":
    main()
