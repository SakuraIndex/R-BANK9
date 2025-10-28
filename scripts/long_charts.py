#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
R-BANK9 long-term charts (7d / 1m / 1y)
- 1d(当日)の生成は行いません（Intraday ワークフローが担当）
- 入力は docs/outputs/rbank9_history.csv を最優先で使用
- 列名が不定でも自動検出し、NaN/重複/同値レンジを安全に処理
- 変動が極端に小さくても線が見えるように y 軸に自動パディング
"""

from __future__ import annotations
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.dates import AutoDateLocator, AutoDateFormatter

INDEX_KEY = os.environ.get("INDEX_KEY", "rbank9")
OUT_DIR   = Path(os.environ.get("OUT_DIR", "docs/outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
#  Logging (控えめに)
def log(msg: str) -> None:
    print(f"[long] {msg}", flush=True)

# ──────────────────────────────────────────────────────────────────────
# ダークテーマ（見やすいコントラスト）
def apply_dark(ax, title: str, y_label: str):
    fig = ax.figure
    fig.set_size_inches(16, 8)
    fig.set_dpi(160)

    bg = "#111317"
    fg = "#e8eef7"
    grid = "#2a2e3a"

    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    for s in ax.spines.values():
        s.set_visible(False)

    ax.grid(True, which="major", linestyle="-", linewidth=0.8, color=grid, alpha=0.5)
    ax.grid(True, which="minor", linestyle="-", linewidth=0.5, color=grid, alpha=0.25)

    ax.tick_params(axis="both", colors=fg, labelsize=10)
    ax.yaxis.label.set_color(fg)
    ax.xaxis.label.set_color(fg)
    ax.title.set_color(fg)

    ax.set_title(title, fontsize=24, fontweight="bold", pad=18)
    ax.set_xlabel("Time", labelpad=10)
    ax.set_ylabel(y_label, labelpad=10)

# ──────────────────────────────────────────────────────────────────────
# 入力読み込み（history 優先）
def _pick_numeric_col(df: pd.DataFrame) -> str | None:
    # “値”らしい列を自動で選ぶ
    candidates = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        non_na = s.notna().sum()
        if non_na >= max(3, len(s)//5):  # 有効値が十分にある
            candidates.append((c, non_na))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]

def load_history() -> pd.DataFrame:
    # 優先: history.csv, 予備: intraday.csv（足りない時）
    hist = OUT_DIR / f"{INDEX_KEY}_history.csv"
    intra = OUT_DIR / f"{INDEX_KEY}_intraday.csv"

    if hist.exists():
        df = pd.read_csv(hist, parse_dates=[0], index_col=0)
        src = hist.name
    elif intra.exists():
        df = pd.read_csv(intra, parse_dates=[0], index_col=0)
        src = intra.name
    else:
        log("no csv found (history/intraday).")
        return pd.DataFrame()

    # 最も “値” らしい列を 1 つ選ぶ
    col = _pick_numeric_col(df)
    if not col:
        log(f"no numeric column found in {src}")
        return pd.DataFrame()

    s = pd.to_numeric(df[col], errors="coerce")
    s.index = pd.to_datetime(df.index, errors="coerce")

    # クレンジング
    s = s.dropna()
    s = s[~s.index.duplicated(keep="last")]
    s = s.sort_index()

    # 年レンジ sanity
    s = s[(s.index.year >= 2000) & (s.index.year <= 2100)]

    out = pd.DataFrame({"ts": s.index, "val": s.values}).reset_index(drop=True)
    log(f"loaded {src}: rows={len(out)}, col={col}")
    return out

# ──────────────────────────────────────────────────────────────────────
# 期間切り出し（level をそのまま描画）
SPAN_DAYS = {"7d": 7, "1m": 30, "1y": 365}

def slice_span(df: pd.DataFrame, span: str) -> pd.DataFrame:
    if df.empty:
        return df
    days = SPAN_DAYS[span]
    last_ts = df["ts"].max()
    start   = last_ts - pd.Timedelta(days=days)
    w = df[df["ts"] >= start].copy()
    # 2点未満なら全体で補完（線が引けないため）
    if len(w) < 2:
        w = df.tail(2).copy()
    return w

# y軸が同値で潰れないようにパディング
def pad_ylim(vals: np.ndarray, ratio: float = 0.05) -> tuple[float, float]:
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return -1.0, 1.0
    if vmin == vmax:
        pad = max(1.0, abs(vmax) * 0.02)
        return vmin - pad, vmax + pad
    span = vmax - vmin
    pad = span * ratio
    return vmin - pad, vmax + pad

# ──────────────────────────────────────────────────────────────────────
# 描画
def plot_level(df: pd.DataFrame, title: str, out_png: Path):
    fig, ax = plt.subplots()
    apply_dark(ax, title, y_label="Level (index)")

    # 実線は見やすい色・太さ
    ax.plot(df["ts"].values, df["val"].values, linewidth=2.4, color="#ff615a")

    # 日付軸
    major = AutoDateLocator(minticks=5, maxticks=10)
    ax.xaxis.set_major_locator(major)
    ax.xaxis.set_major_formatter(AutoDateFormatter(major))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))

    # y パディング
    ymin, ymax = pad_ylim(df["val"].values, ratio=0.06)
    ax.set_ylim(ymin, ymax)

    fig.tight_layout()
    fig.savefig(out_png, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    log(f"WROTE: {out_png.name}")

# ──────────────────────────────────────────────────────────────────────
def main():
    df_all = load_history()
    if df_all.empty:
        log("no data -> skip")
        return

    for span in ["7d", "1m", "1y"]:
        d = slice_span(df_all, span)
        if d.empty or len(d) < 2:
            log(f"skip {span}: not enough data")
            continue
        title = f"{INDEX_KEY.upper()} ({span} level)"
        out = OUT_DIR / f"{INDEX_KEY}_{span}.png"
        plot_level(d, title, out)

    # 実行記録
    (OUT_DIR / "_last_run.txt").write_text(
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        encoding="utf-8",
    )

if __name__ == "__main__":
    main()
