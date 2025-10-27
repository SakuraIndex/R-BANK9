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

# ％計算の安全パラメータ
EPS        = 0.2     # 極小分母回避（level≒0を避ける）
CLAMP_PCT  = 30.0    # 表示用に±30%へクランプ（外れ値で見栄えが壊れないように）

# ─────────────────────────────────────────────────────────────
#  見た目：1日の intraday と同水準のダークテーマ（白文字・薄グリッド）
# ─────────────────────────────────────────────────────────────
def apply_dark_theme(fig, ax):
    fig.set_size_inches(16, 8)
    fig.set_dpi(200)  # 高精細
    fig.patch.set_facecolor("#111317")
    ax.set_facecolor("#111317")

    for sp in ax.spines.values():
        sp.set_visible(False)

    ax.tick_params(axis="both", colors="#ffffff", labelsize=10)
    ax.yaxis.label.set_color("#ffffff")
    ax.xaxis.label.set_color("#ffffff")
    ax.title.set_color("#ffffff")

    # グリッド（major/minor 両方）
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.18, color="#ffffff")
    ax.grid(True, which="minor", linestyle="-", linewidth=0.4, alpha=0.10, color="#ffffff")

# ─────────────────────────────────────────────────────────────
#  入力CSVの選択：span専用 → history → 1d（存在するものを使う）
# ─────────────────────────────────────────────────────────────
def pick_csv(span: str) -> Path:
    # span 専用（7d/1m/1y）があれば最優先
    if span in ("7d", "1m", "1y"):
        c = OUT_DIR / f"{INDEX_KEY}_{span}.csv"
        if c.exists():
            return c
        h = OUT_DIR / f"{INDEX_KEY}_history.csv"
        if h.exists():
            return h
    # 最後の砦
    return OUT_DIR / f"{INDEX_KEY}_1d.csv"

def load_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        print(f"[long] CSV not found: {csv_path}")
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        print(f"[long] CSV needs >=2 columns: {csv_path}")
        return pd.DataFrame()
    ts_col, val_col = df.columns[:2]
    df = df.rename(columns={ts_col: "ts", val_col: "val"})
    df["ts"] = pd.to_datetime(df["ts"], utc=False, errors="coerce")
    df = df.dropna(subset=["ts", "val"]).sort_values("ts").reset_index(drop=True)
    return df

# ─────────────────────────────────────────────────────────────
#  1日の安定ベース：10:00以降の最初の有効値 → なければ最初の有効値
# ─────────────────────────────────────────────────────────────
def stable_baseline(df_day: pd.DataFrame) -> float | None:
    if df_day.empty:
        return None
    mask = (df_day["ts"].dt.hour > 10) | ((df_day["ts"].dt.hour == 10) & (df_day["ts"].dt.minute >= 0))
    cand = df_day.loc[mask & (df_day["val"].abs() >= EPS)]
    if not cand.empty:
        return float(cand.iloc[0]["val"])
    cand2 = df_day.loc[df_day["val"].abs() >= EPS]
    if not cand2.empty:
        return float(cand2.iloc[0]["val"])
    return float(df_day.iloc[0]["val"])

def clamp_pct(p: float) -> float:
    if p >  CLAMP_PCT: return CLAMP_PCT
    if p < -CLAMP_PCT: return -CLAMP_PCT
    return p

def calc_sane_pct(base: float, close: float) -> float:
    denom = max(abs(base), EPS)
    return clamp_pct((close - base) / denom * 100.0)

# ─────────────────────────────────────────────────────────────
#  スパン別に％系列を生成
# ─────────────────────────────────────────────────────────────
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

    # 7d/1m/1y は期間先頭の値を基準に％算出
    last = df["ts"].max()
    days_map = {"7d": 7, "1m": 30, "1y": 365}
    days = days_map.get(span, 7)
    df_span = df[df["ts"] >= (last - pd.Timedelta(days=days))].copy()
    if df_span.empty:
        return df_span
    base = float(df_span.iloc[0]["val"])
    df_span["pct"] = df_span["val"].apply(lambda x: calc_sane_pct(base, x))
    return df_span

# ─────────────────────────────────────────────────────────────
#  描画
# ─────────────────────────────────────────────────────────────
def plot_one_span(df_pct: pd.DataFrame, title: str, out_png: Path):
    fig, ax = plt.subplots()
    apply_dark_theme(fig, ax)

    ax.set_title(title, fontsize=26, fontweight="bold", pad=18)
    ax.set_xlabel("Time", labelpad=10)
    ax.set_ylabel("Change (%)", labelpad=10)

    ax.plot(df_pct["ts"].values, df_pct["pct"].values, linewidth=2.6)

    major = AutoDateLocator(minticks=5, maxticks=10)
    ax.xaxis.set_major_locator(major)
    ax.xaxis.set_major_formatter(AutoDateFormatter(major))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.yaxis.set_minor_locator(MaxNLocator(nbins=50))

    fig.tight_layout()
    fig.savefig(out_png, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"[long] WROTE: {out_png}")

# ─────────────────────────────────────────────────────────────
#  main
# ─────────────────────────────────────────────────────────────
def main():
    spans = ["1d", "7d", "1m", "1y"]
    for span in spans:
        csv = pick_csv(span)
        df  = load_csv(csv)
        if df.empty:
            print(f"[long] Skip {span}: empty {csv}")
            continue
        df_pct = make_pct_series(df, span)
        if df_pct.empty or "pct" not in df_pct:
            print(f"[long] Skip {span}: no pct series")
            continue
        out_png = OUT_DIR / f"{INDEX_KEY}_{span}.png"
        title   = f"{INDEX_KEY.upper()} ({span})"
        plot_one_span(df_pct, title, out_png)

if __name__ == "__main__":
    main()
