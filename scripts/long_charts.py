#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.dates import AutoDateLocator, AutoDateFormatter
from datetime import datetime, timezone

INDEX_KEY = os.environ.get("INDEX_KEY", "rbank9")
OUT_DIR   = Path(os.environ.get("OUT_DIR", "docs/outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== params =====
EPS = 1e-9        # ほぼゼロ除算回避
USE_CLAMP = False # True なら下の CLAMP_PCT を使用
CLAMP_PCT = 60.0

def diag(msg: str):
    print(f"[long] {msg}", flush=True)

# ===== styling =====
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

# ===== data pickers =====
def pick_csv_for_span(span: str) -> Path | None:
    """ spanごとの優先データ源を返す """
    intraday = OUT_DIR / f"{INDEX_KEY}_intraday.csv"
    history  = OUT_DIR / f"{INDEX_KEY}_history.csv"
    one      = OUT_DIR / f"{INDEX_KEY}_1d.csv"

    if span == "1d":
        if intraday.exists(): return intraday
        if one.exists():      return one
        if history.exists():  return history
        return None

    # 7d/1m/1y
    if history.exists():  return history
    if intraday.exists(): return intraday
    if one.exists():      return one
    return None

def _coerce_datetime(s: pd.Series) -> pd.Series:
    # 文字列/数値どちらでも頑張って時刻にする
    ts = pd.to_datetime(s, errors="coerce", utc=False)
    # もし全滅なら、UNIX秒/ミリ秒を疑って数値化→再トライ
    if ts.notna().sum() == 0:
        num = pd.to_numeric(s, errors="coerce")
        ts = pd.to_datetime(num, errors="coerce", unit="s")
        if ts.notna().sum() == 0:
            ts = pd.to_datetime(num, errors="coerce", unit="ms")
    return ts

def load_csv_any(csv_path: Path) -> pd.DataFrame:
    """1列目=時刻、2列目=値 という一般形/未知ヘッダでも動くロバストローダー"""
    if not csv_path or not csv_path.exists():
        return pd.DataFrame()
    df_raw = pd.read_csv(csv_path)
    if df_raw.empty or df_raw.shape[1] < 2:
        return pd.DataFrame()

    # 時刻列を探す（ts/time/timestamp/…優先 → 先頭列）
    low = [c.lower().strip() for c in df_raw.columns]
    ts_candidates = []
    for pref in ("ts", "time", "timestamp", "date", "datetime"):
        if pref in low:
            ts_candidates.append(df_raw.columns[low.index(pref)])
    if not ts_candidates:
        ts_candidates = [df_raw.columns[0]]

    ts = _coerce_datetime(df_raw[ts_candidates[0]])

    # 値列を決める：ts列以外から NaN 以外が多い列
    val_col = None
    best_ok = -1
    for c in df_raw.columns:
        if c == ts_candidates[0]:
            continue
        v = pd.to_numeric(df_raw[c], errors="coerce")
        ok = v.notna().sum()
        if ok > best_ok:
            best_ok = ok
            val_col = c
    if val_col is None:
        return pd.DataFrame()

    val = pd.to_numeric(df_raw[val_col], errors="coerce")
    df = pd.DataFrame({"ts": ts, "val": val}).dropna().sort_values("ts")
    # 年の異常値を除去
    df = df[(df["ts"].dt.year >= 2000) & (df["ts"].dt.year <= 2100)]
    df = df[~df["ts"].duplicated(keep="last")].reset_index(drop=True)

    return df

# ===== pct helpers =====
def clamp_pct(p: float) -> float:
    if not USE_CLAMP:
        return p
    if p >  CLAMP_PCT: return CLAMP_PCT
    if p < -CLAMP_PCT: return -CLAMP_PCT
    return p

def calc_pct(base: float, x: float) -> float:
    denom = max(abs(base), EPS)
    return clamp_pct((x - base) / denom * 100.0)

def stable_baseline_1d(df_day: pd.DataFrame) -> float | None:
    """10:00以降の最初の非ゼロ/非極小を基準。なければ最初の有効値"""
    if df_day.empty:
        return None
    h = df_day["ts"].dt.hour
    m = df_day["ts"].dt.minute
    after_10 = (h > 10) | ((h == 10) & (m >= 0))
    cand = df_day.loc[after_10 & (df_day["val"].abs() > EPS)]
    if not cand.empty:
        return float(cand.iloc[0]["val"])
    cand2 = df_day.loc[df_day["val"].abs() > EPS]
    if not cand2.empty:
        return float(cand2.iloc[0]["val"])
    return float(df_day.iloc[0]["val"])

def make_pct_series(df: pd.DataFrame, span: str) -> pd.DataFrame:
    if df.empty:
        return df

    if span == "1d":
        the_day = df["ts"].dt.floor("D").iloc[-1]
        d = df[df["ts"].dt.floor("D") == the_day].copy()
        if d.empty:
            return d
        base = stable_baseline_1d(d)
        if base is None:
            return pd.DataFrame()
        d["pct"] = d["val"].apply(lambda v: calc_pct(base, v))
        return d

    # 7d/1m/1y: 期間先頭の値を基準
    last = df["ts"].max()
    days = {"7d": 7, "1m": 30, "1y": 365}.get(span, 7)
    w = df[df["ts"] >= (last - pd.Timedelta(days=days))].copy()
    if w.empty:
        return w
    base = float(w.iloc[0]["val"])
    w["pct"] = w["val"].apply(lambda v: calc_pct(base, v))
    return w

# ===== plotting =====
def plot_span(dfp: pd.DataFrame, title: str, out_png: Path, ylabel="Change (%)"):
    fig, ax = plt.subplots()
    apply_dark_theme(fig, ax)
    ax.set_title(title, fontsize=26, fontweight="bold", pad=18)
    ax.set_xlabel("Time", labelpad=10)
    ax.set_ylabel(ylabel, labelpad=10)
    ax.plot(dfp["ts"].values, dfp["pct"].values, linewidth=2.6, color="#ff615a")
    major = AutoDateLocator(minticks=5, maxticks=10)
    ax.xaxis.set_major_locator(major)
    ax.xaxis.set_major_formatter(AutoDateFormatter(major))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.yaxis.set_minor_locator(MaxNLocator(nbins=50))
    fig.tight_layout()
    fig.savefig(out_png, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    diag(f"WROTE {out_png.name}  rows={len(dfp)}  min={dfp['pct'].min():.3f}  max={dfp['pct'].max():.3f}")

# ===== main =====
def main():
    wrote_any = False
    for span in ["1d", "7d", "1m", "1y"]:
        csv = pick_csv_for_span(span)
        if not csv:
            diag(f"Skip {span}: no source csv")
            continue
        df = load_csv_any(csv)
        diag(f"Source for {span}: {csv.name}  rows={len(df)}")

        if df.empty or ("ts" not in df) or ("val" not in df):
            diag(f"Skip {span}: invalid dataframe")
            continue

        dfp = make_pct_series(df, span)
        if dfp.empty or ("pct" not in dfp):
            diag(f"Skip {span}: no pct series")
            continue

        out_png = OUT_DIR / f"{INDEX_KEY}_{span}.png"
        plot_span(dfp, f"{INDEX_KEY.upper()} ({span})", out_png)
        wrote_any = True

    # マーカー（実行時刻を書くだけでも良い）
    (OUT_DIR / "_last_run.txt").write_text(
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        encoding="utf-8",
    )
    if not wrote_any:
        diag("No chart written.")

if __name__ == "__main__":
    main()
