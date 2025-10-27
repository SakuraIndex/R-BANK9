#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import pandas as pd
import numpy as np
import hashlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.dates import AutoDateLocator, AutoDateFormatter

INDEX_KEY = os.environ.get("INDEX_KEY", "rbank9")
OUT_DIR   = Path(os.environ.get("OUT_DIR", "docs/outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPS       = 0.2
CLAMP_PCT = 30.0

def apply_dark_theme(fig, ax):
    fig.set_size_inches(16, 8)
    fig.set_dpi(200)
    fig.patch.set_facecolor("#111317")
    ax.set_facecolor("#111317")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(axis="both", colors="#ffffff", labelsize=10)
    ax.yaxis.label.set_color("#ffffff")
    ax.xaxis.label.set_color("#ffffff")
    ax.title.set_color("#ffffff")
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.18, color="#ffffff")
    ax.grid(True, which="minor", linestyle="-", linewidth=0.4, alpha=0.10, color="#ffffff")

def sha256(p: Path) -> str:
    if not p.exists(): return "-"
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]

def choose_csv(span: str) -> Path | None:
    # 候補を“存在→十分な有効行数(>=10)”の順で選ぶ
    cands = []
    if span in ("7d","1m","1y"):
        cands += [OUT_DIR / f"{INDEX_KEY}_{span}.csv"]
        cands += [OUT_DIR / f"{INDEX_KEY}_history.csv"]
    # 常に最後に 1d をフォールバック
    cands += [OUT_DIR / f"{INDEX_KEY}_1d.csv", OUT_DIR / f"{INDEX_KEY}_intraday.csv"]

    for c in cands:
        if not c.exists(): continue
        try:
            df = pd.read_csv(c)
            if df.shape[1] < 2: continue
            # “十分”の定義：非欠損が 10 行以上
            nonnull = df.iloc[:, :2].dropna().shape[0]
            if nonnull >= 10:
                print(f"[pick] span={span} -> {c} (rows={len(df)})")
                return c
            else:
                print(f"[pick] skip {c} (nonnull<{10})")
        except Exception as e:
            print(f"[pick] read error {c}: {e}")
    print(f"[pick] no usable csv for span={span}")
    return None

def best_datetime_column(df: pd.DataFrame) -> str | None:
    hints = ["ts","time","timestamp","date","datetime"]
    low = [c.lower().strip() for c in df.columns]
    for h in hints:
        if h in low:
            c = df.columns[low.index(h)]
            s = pd.to_datetime(df[c], errors="coerce", utc=False)
            if s.notna().sum() >= max(5, len(s)//4):
                df[c] = s
                return c
    best_c, best_ok = None, -1
    for c in df.columns:
        s = pd.to_datetime(df[c], errors="coerce", utc=False)
        if s.notna().sum() > best_ok:
            best_c, best_ok = c, s.notna().sum()
            df[c] = s
    return best_c

def best_value_column(df: pd.DataFrame, ts_col: str) -> str | None:
    cand = []
    for c in df.columns:
        if c == ts_col: continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= max(5, len(s)//4) and (s.std(skipna=True) > 1e-9):
            cand.append((c, s.notna().sum()))
            df[c] = s
    if not cand: return None
    cand.sort(key=lambda x: x[1], reverse=True)
    return cand[0][0]

def load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty or df.shape[1] < 2:
        return pd.DataFrame()
    ts_col  = best_datetime_column(df.copy())
    if not ts_col: return pd.DataFrame()
    val_col = best_value_column(df.copy(), ts_col)
    if not val_col: return pd.DataFrame()

    out = pd.DataFrame({
        "ts":  pd.to_datetime(df[ts_col], errors="coerce", utc=False),
        "val": pd.to_numeric(df[val_col], errors="coerce")
    }).dropna().sort_values("ts").reset_index(drop=True)

    out = out[~out["ts"].duplicated(keep="last")]
    mask = (out["ts"].dt.year >= 2000) & (out["ts"].dt.year <= 2100)
    out = out.loc[mask].reset_index(drop=True)
    return out

def clamp_pct(p: float) -> float:
    if p >  CLAMP_PCT: return CLAMP_PCT
    if p < -CLAMP_PCT: return -CLAMP_PCT
    return p

def calc_sane_pct(base: float, close: float) -> float:
    denom = max(abs(base), EPS)
    return clamp_pct((close - base) / denom * 100.0)

def stable_baseline(df_day: pd.DataFrame) -> float | None:
    if df_day.empty: return None
    mask = (df_day["ts"].dt.hour > 10) | ((df_day["ts"].dt.hour == 10) & (df_day["ts"].dt.minute >= 0))
    cand = df_day.loc[mask & (df_day["val"].abs() >= EPS)]
    if not cand.empty: return float(cand.iloc[0]["val"])
    cand2 = df_day.loc[df_day["val"].abs() >= EPS]
    if not cand2.empty: return float(cand2.iloc[0]["val"])
    return float(df_day.iloc[0]["val"])

def make_pct_series(df: pd.DataFrame, span: str) -> pd.DataFrame:
    if df.empty: return df
    if span == "1d":
        day = df["ts"].dt.floor("D").iloc[-1]
        d = df[df["ts"].dt.floor("D") == day].copy()
        if d.empty: return d
        base = stable_baseline(d)
        if base is None: return pd.DataFrame()
        d["pct"] = d["val"].apply(lambda v: calc_sane_pct(base, v))
        return d
    last = df["ts"].max()
    days = {"7d":7, "1m":30, "1y":365}.get(span, 7)
    w = df[df["ts"] >= (last - pd.Timedelta(days=days))].copy()
    if w.empty: return w
    base = float(w.iloc[0]["val"])
    w["pct"] = w["val"].apply(lambda v: calc_sane_pct(base, v))
    return w

def plot_one_span(dfp: pd.DataFrame, title: str, out_png: Path):
    before = sha256(out_png)
    fig, ax = plt.subplots()
    apply_dark_theme(fig, ax)
    ax.set_title(title, fontsize=26, fontweight="bold", pad=18)
    ax.set_xlabel("Time", labelpad=10)
    ax.set_ylabel("Change (%)", labelpad=10)
    ax.plot(dfp["ts"].values, dfp["pct"].values, linewidth=2.6, color="#ff615a")
    major = AutoDateLocator(minticks=5, maxticks=10)
    ax.xaxis.set_major_locator(major)
    ax.xaxis.set_major_formatter(AutoDateFormatter(major))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.yaxis.set_minor_locator(MaxNLocator(nbins=50))
    fig.tight_layout()
    fig.savefig(out_png, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    after = sha256(out_png)
    size  = out_png.stat().st_size if out_png.exists() else 0
    print(f"[write] {out_png.name}: sha {before} -> {after}  size={size/1024:.1f}KB")

def main():
    spans = ["1d","7d","1m","1y"]
    for span in spans:
        csv = choose_csv(span)
        if not csv:
            print(f"[main] Skip {span}: no csv")
            continue
        df = load_csv(csv)
        if df.empty:
            print(f"[main] Skip {span}: empty df from {csv}")
            continue
        dfp = make_pct_series(df, span)
        if dfp.empty or "pct" not in dfp:
            print(f"[main] Skip {span}: no pct series")
            continue
        out_png = OUT_DIR / f"{INDEX_KEY}_{span}.png"
        plot_one_span(dfp, f"{INDEX_KEY.upper()} ({span})", out_png)

if __name__ == "__main__":
    main()
