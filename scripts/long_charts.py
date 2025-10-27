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
EPS       = 0.2      # avoid div by near-0 when making % from level
CLAMP_PCT = 30.0     # visual clamp for extreme spikes
FIGSIZE   = (16, 8)
DPI       = 200

# ================= theme (match intraday quality) =================
def apply_dark_theme(fig, ax):
    fig.set_size_inches(*FIGSIZE)
    fig.set_dpi(DPI)
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

# ================= CSV pick =================
def pick_csv(span: str) -> Path:
    """7d/1m/1y は history を最優先。無ければ既存フォールバック。"""
    if span in ("7d", "1m", "1y"):
        h = OUT_DIR / f"{INDEX_KEY}_history.csv"
        if h.exists():
            return h
        # next best: span 専用 CSV（あるなら）
        s = OUT_DIR / f"{INDEX_KEY}_{span}.csv"
        if s.exists():
            return s
    # default（1d など）
    return OUT_DIR / f"{INDEX_KEY}_1d.csv"

# ================= robust loader =================
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
        ok = s.notna().sum()
        if ok > best_ok:
            best_c, best_ok = c, ok
            df[c] = s
    return best_c

def best_value_column(df: pd.DataFrame, ts_col: str) -> str | None:
    cand = []
    for c in df.columns:
        if c == ts_col:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= max(5, len(s)//4) and (s.std(skipna=True) > 1e-12):
            cand.append((c, s.notna().sum()))
            df[c] = s
    if not cand:
        return None
    cand.sort(key=lambda x: x[1], reverse=True)
    return cand[0][0]

def load_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        print(f"[long] CSV not found: {csv_path}")
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    if df.empty or df.shape[1] < 2:
        print(f"[long] CSV invalid shape: {csv_path}")
        return pd.DataFrame()

    # datetime & value columns
    df_copy = df.copy()
    ts_col = best_datetime_column(df_copy)
    if not ts_col:
        print(f"[long] No datetime-like column in {csv_path}")
        return pd.DataFrame()
    val_col = best_value_column(df_copy, ts_col)
    if not val_col:
        print(f"[long] No numeric value column in {csv_path}")
        return pd.DataFrame()

    out = pd.DataFrame({
        "ts":  pd.to_datetime(df_copy[ts_col], errors="coerce", utc=False),
        "val": pd.to_numeric(df_copy[val_col], errors="coerce")
    }).dropna().sort_values("ts").reset_index(drop=True)

    # unique by ts
    out = out[~out["ts"].duplicated(keep="last")]

    # year sanity
    mask = (out["ts"].dt.year >= 2000) & (out["ts"].dt.year <= 2100)
    out = out.loc[mask].reset_index(drop=True)

    # 付帯情報（列名で％ヒント）
    out.attrs["val_col_name"] = str(val_col)
    return out

# ================= pct helpers =================
def clamp_pct(p: float) -> float:
    if p >  CLAMP_PCT: return CLAMP_PCT
    if p < -CLAMP_PCT: return -CLAMP_PCT
    return p

def is_percent_series(values: pd.Series, colname: str) -> bool:
    """列名や値レンジから“既に％”と推定。"""
    name = (colname or "").lower()
    if any(k in name for k in ["pct", "percent", "%", "change"]):
        return True
    v = values.dropna()
    if v.empty:
        return False
    # 値の典型レンジが±20%以内で平均が 0 近辺なら％とみなす
    if (v.abs().quantile(0.98) <= 20.0) and (abs(v.median()) <= 5.0):
        return True
    return False

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

def make_pct_series(df: pd.DataFrame, span: str) -> pd.DataFrame:
    if df.empty:
        return df

    colname = df.attrs.get("val_col_name", "")
    already_pct = is_percent_series(df["val"], colname)

    if span == "1d":
        the_day = df["ts"].dt.floor("D").iloc[-1]
        d = df[df["ts"].dt.floor("D") == the_day].copy()
        if d.empty:
            return d
        if already_pct:
            d["pct"] = d["val"].clip(-CLAMP_PCT, CLAMP_PCT)
            return d
        base = stable_baseline(d)
        if base is None:
            return pd.DataFrame()
        d["pct"] = d["val"].apply(lambda v: clamp_pct((v - base) / max(abs(base), EPS) * 100.0))
        return d

    # 7d/1m/1y
    last = df["ts"].max()
    days = {"7d": 7, "1m": 30, "1y": 365}.get(span, 7)
    w = df[df["ts"] >= (last - pd.Timedelta(days=days))].copy()
    if w.empty:
        return w
    if already_pct:
        w["pct"] = w["val"].clip(-CLAMP_PCT, CLAMP_PCT)
        return w
    base = float(w.iloc[0]["val"])
    w["pct"] = w["val"].apply(lambda v: clamp_pct((v - base) / max(abs(base), EPS) * 100.0))
    return w

# ================= plotting =================
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

    # y 範囲はデータ±余白、ただし ±CLAMP_PCT 内に収める
    p = dfp["pct"].dropna()
    if not p.empty:
        lo = max(-CLAMP_PCT, float(p.min()) - 2.0)
        hi = min( CLAMP_PCT, float(p.max()) + 2.0)
        if lo >= hi:
            lo, hi = -1.0, 1.0
        ax.set_ylim(lo, hi)

    fig.tight_layout()
    fig.savefig(out_png, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"[long] WROTE: {out_png}")

# ================= main =================
def main():
    spans = ["1d", "7d", "1m", "1y"]
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
        plot_one_span(dfp, f"{INDEX_KEY.upper()} ({span})", OUT_DIR / f"{INDEX_KEY}_{span}.png")

if __name__ == "__main__":
    main()
