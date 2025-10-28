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

# ---- safety for % ----
EPS       = 0.2     # avoid div by near-0
CLAMP_PCT = 30.0    # visual clamp for outliers

# ========== theme (match intraday look) ==========
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

# ========== robust CSV loader ==========
def _pick_ts_col(df: pd.DataFrame) -> str | None:
    # prefer name hints
    hints = ["ts","time","timestamp","date","datetime"]
    low = [c.strip().lower() for c in df.columns]
    for h in hints:
        if h in low:
            c = df.columns[low.index(h)]
            s = pd.to_datetime(df[c], errors="coerce", utc=False)
            if s.notna().sum() >= max(5, len(s)//4):
                df[c] = s
                return c
    # fallback: best convertible
    best, best_ok = None, -1
    for c in df.columns:
        s = pd.to_datetime(df[c], errors="coerce", utc=False)
        ok = s.notna().sum()
        if ok > best_ok:
            best, best_ok = c, ok
            df[c] = s
    return best

def _pick_val_col(df: pd.DataFrame, ts_col: str) -> str | None:
    cand = []
    for c in df.columns:
        if c == ts_col: continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= max(5, len(s)//4) and (s.std(skipna=True) > 0):
            df[c] = s
            cand.append((c, s.notna().sum()))
    if not cand: return None
    cand.sort(key=lambda x: x[1], reverse=True)
    return cand[0][0]

def _load_generic(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    if df.empty or df.shape[1] < 2:
        return pd.DataFrame()
    ts_col  = _pick_ts_col(df)
    if not ts_col: return pd.DataFrame()
    val_col = _pick_val_col(df, ts_col)
    if not val_col: return pd.DataFrame()
    out = pd.DataFrame({
        "ts":  pd.to_datetime(df[ts_col], errors="coerce", utc=False),
        "val": pd.to_numeric(df[val_col], errors="coerce")
    }).dropna().sort_values("ts").reset_index(drop=True)
    # keep sane years
    if not out.empty and "ts" in out:
        mask = (out["ts"].dt.year >= 2000) & (out["ts"].dt.year <= 2100)
        out = out.loc[mask].reset_index(drop=True)
    return out

def load_intraday_df() -> pd.DataFrame:
    # 優先順位: *_1d.csv → *_intraday.csv
    c1 = OUT_DIR / f"{INDEX_KEY}_1d.csv"
    c2 = OUT_DIR / f"{INDEX_KEY}_intraday.csv"
    df = _load_generic(c1)
    if df.empty:
        df = _load_generic(c2)
    return df

def load_history_df() -> pd.DataFrame:
    # 7d/1m/1y は必ず history を使う
    ch = OUT_DIR / f"{INDEX_KEY}_history.csv"
    return _load_generic(ch)

# ========== pct helpers ==========
def _clamp(p: float) -> float:
    if p >  CLAMP_PCT: return CLAMP_PCT
    if p < -CLAMP_PCT: return -CLAMP_PCT
    return p

def _pct(base: float, v: float) -> float:
    denom = max(abs(base), EPS)
    return _clamp((v - base) / denom * 100.0)

def _stable_baseline(df_day: pd.DataFrame) -> float | None:
    if df_day.empty: return None
    mask = (df_day["ts"].dt.hour > 10) | ((df_day["ts"].dt.hour == 10) & (df_day["ts"].dt.minute >= 0))
    cand = df_day.loc[mask & (df_day["val"].abs() >= EPS)]
    if not cand.empty: return float(cand.iloc[0]["val"])
    cand2 = df_day.loc[df_day["val"].abs() >= EPS]
    if not cand2.empty: return float(cand2.iloc[0]["val"])
    return float(df_day.iloc[0]["val"])

def make_pct_series_1d(intra: pd.DataFrame) -> pd.DataFrame:
    if intra.empty: return intra
    day = intra["ts"].dt.floor("D").iloc[-1]
    d   = intra[intra["ts"].dt.floor("D") == day].copy()
    if d.empty: return d
    base = _stable_baseline(d)
    if base is None: return pd.DataFrame()
    d["pct"] = d["val"].apply(lambda v: _pct(base, v))
    return d

def make_pct_series_history(hist: pd.DataFrame, span: str) -> pd.DataFrame:
    if hist.empty: return hist
    last = hist["ts"].max()
    days = {"7d": 7, "1m": 30, "1y": 365}[span]
    w = hist[hist["ts"] >= (last - pd.Timedelta(days=days))].copy()
    if w.empty: return w
    base = float(w.iloc[0]["val"])
    w["pct"] = w["val"].apply(lambda v: _pct(base, v))
    return w

# ========== plotting ==========
def plot_one_span(dfp: pd.DataFrame, title: str, out_png: Path):
    if dfp.empty or "pct" not in dfp:
        return
    fig, ax = plt.subplots()
    apply_dark_theme(fig, ax)
    ax.set_title(title, fontsize=26, fontweight="bold", pad=18)
    ax.set_xlabel("Time", labelpad=10)
    ax.set_ylabel("Change (%)", labelpad=10)
    # 線色を明示（ダーク背景で確実に見える）
    ax.plot(dfp["ts"].values, dfp["pct"].values, linewidth=2.6, color="#00d4ff")
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
    intra = load_intraday_df()
    hist  = load_history_df()

    # 1d → intraday / 7d,1m,1y → history
    spans = ["1d","7d","1m","1y"]
    for span in spans:
        if span == "1d":
            dfp = make_pct_series_1d(intra)
        else:
            dfp = make_pct_series_history(hist, span)
        if dfp is None or dfp.empty or "pct" not in dfp:
            print(f"[long] Skip {span}: empty or no pct")
            continue
        out_png = OUT_DIR / f"{INDEX_KEY}_{span}.png"
        plot_one_span(dfp, f"{INDEX_KEY.upper()} ({span})", out_png)

if __name__ == "__main__":
    main()
