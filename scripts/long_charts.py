#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.dates import AutoDateLocator, AutoDateFormatter

INDEX_KEY = os.environ.get("INDEX_KEY", "rbank9")
OUT_DIR   = Path(os.environ.get("OUT_DIR", "docs/outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- safety for % ----
EPS       = 0.2     # avoid div by near-0
CLAMP_PCT = 30.0    # visual clamp

# ========== theme (match intraday quality) ==========
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

# ========== CSV pick ==========
def pick_csv(span: str) -> Path:
    if span in ("7d","1m","1y"):
        c = OUT_DIR / f"{INDEX_KEY}_{span}.csv"
        if c.exists(): return c
        h = OUT_DIR / f"{INDEX_KEY}_history.csv"
        if h.exists(): return h
    return OUT_DIR / f"{INDEX_KEY}_1d.csv"

# ========== robust loader ==========
def best_datetime_column(df: pd.DataFrame) -> str | None:
    # 1) name-based hints
    hints = ["ts","time","timestamp","date","datetime"]
    low = [c.lower().strip() for c in df.columns]
    for h in hints:
        if h in low:
            c = df.columns[low.index(h)]
            s = pd.to_datetime(df[c], errors="coerce", utc=False)
            ok = s.notna().sum()
            if ok >= max(5, len(s)//4):  # enough valid
                df[c] = s
                return c
    # 2) try-all: pick column with max valid datetimes within sane year
    best_c, best_ok = None, -1
    for c in df.columns:
        s = pd.to_datetime(df[c], errors="coerce", utc=False)
        if s.notna().sum() > best_ok:
            best_c, best_ok = c, s.notna().sum()
            df[c] = s
    if best_c is None: return None
    return best_c

def best_value_column(df: pd.DataFrame, ts_col: str) -> str | None:
    cand = []
    for c in df.columns:
        if c == ts_col: continue
        s = pd.to_numeric(df[c], errors="coerce")
        # reject near-constant zero/NaN columns
        if s.notna().sum() >= max(5, len(s)//4) and (s.std(skipna=True) > 1e-9):
            cand.append((c, s.notna().sum()))
            df[c] = s
    if not cand: return None
    # choose most-complete
    cand.sort(key=lambda x: x[1], reverse=True)
    return cand[0][0]

def load_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        print(f"[long] CSV not found: {csv_path}"); return pd.DataFrame()
    df = pd.read_csv(csv_path)
    if df.empty or df.shape[1] < 2:
        print(f"[long] CSV invalid shape: {csv_path}"); return pd.DataFrame()

    ts_col = best_datetime_column(df.copy())
    if not ts_col:
        print(f"[long] No datetime-like column in {csv_path}"); return pd.DataFrame()
    val_col = best_value_column(df.copy(), ts_col)
    if not val_col:
        print(f"[long] No numeric value column in {csv_path}"); return pd.DataFrame()

    # finalize
    out = pd.DataFrame({
        "ts":  pd.to_datetime(df[ts_col], errors="coerce", utc=False),
        "val": pd.to_numeric(df[val_col], errors="coerce")
    }).dropna().sort_values("ts").reset_index(drop=True)

    # drop duplicates by ts
    out = out[~out["ts"].duplicated(keep="last")]

    # sanity: restrict to reasonable years (2000-2100)
    mask = (out["ts"].dt.year >= 2000) & (out["ts"].dt.year <= 2100)
    out = out.loc[mask].reset_index(drop=True)

    return out

# ========== pct helpers ==========
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
        the_day = df["ts"].dt.floor("D").iloc[-1]
        d = df[df["ts"].dt.floor("D") == the_day].copy()
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

# ========== plotting ==========
def plot_one_span(dfp: pd.DataFrame, title: str, out_png: Path):
    fig, ax = plt.subplots()
    apply_dark_theme(fig, ax)
    ax.set_title(title, fontsize=26, fontweight="bold", pad=18)
    ax.set_xlabel("Time", labelpad=10)
    ax.set_ylabel("Change (%)", labelpad=10)
    ax.plot(dfp["ts"].values, dfp["pct"].values, linewidth=2.6)
    major = AutoDateLocator(minticks=5, maxticks=10)
    ax.xaxis.set_major_locator(major)
    ax.xaxis.set_major_formatter(AutoDateFormatter(major))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.yaxis.set_minor_locator(MaxNLocator(nbins=50))
    fig.tight_layout()
    fig.savefig(out_png, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"[long] WROTE: {out_png}")

# ========== main ==========
def main():
    spans = ["1d","7d","1m","1y"]
    for span in spans:
        csv = pick_csv(span)
        df  = load_csv(csv)
        if df.empty:
            print(f"[long] Skip {span}: empty/invalid {csv}")
            continue
        dfp = make_pct_series(df, span)
        if dfp.empty or "pct" not in dfp:
            print(f"[long] Skip {span}: no pct series")
            continue
        out_png = OUT_DIR / f"{INDEX_KEY}_{span}.png"
        plot_one_span(dfp, f"{INDEX_KEY.upper()} ({span})", out_png)

if __name__ == "__main__":
    main()
