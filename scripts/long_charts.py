# scripts/long_charts.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re
import pandas as pd
import pytz
import matplotlib
import matplotlib.pyplot as plt

JP = pytz.timezone("Asia/Tokyo")
NY = pytz.timezone("America/New_York")

BG, FG, TITLE = "#0E1117", "#E6E6E6", "#f2b6c6"
UP, DOWN, GRID_A = "#3bd6c6", "#ff6b6b", 0.25

matplotlib.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": FG,
    "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
    "text.color": FG, "grid.color": FG, "savefig.facecolor": BG,
})

OUTDIR = os.path.join("docs", "outputs")

def _lower(df): df.columns=[str(c).strip().lower() for c in df.columns]; return df

def _find_time_col(cols):
    for c in cols:
        if re.search(r"time|日時|date|datetime|timestamp|時刻", c): return c
    return cols[0] if cols else None

def read_intraday(path:str)->pd.DataFrame:
    if not os.path.exists(path): return pd.DataFrame(columns=["time","value"])
    raw = pd.read_csv(path, dtype=str)
    if raw.empty: return pd.DataFrame(columns=["time","value"])
    df = _lower(raw)
    tcol = _find_time_col(df.columns)
    if tcol is None: return pd.DataFrame(columns=["time","value"])

    # 値カラム推定
    vcol=None
    for c in df.columns:
        if c==tcol: continue
        if any(k in c for k in ["value","index","score","mean"]): vcol=c;break
    if vcol is None:
        for c in df.columns:
            if c==tcol: continue
            try: pd.to_numeric(df[c]); vcol=c; break
            except: pass

    # 時刻 → tz-aware
    t = pd.to_datetime(df[tcol], errors="coerce", utc=True)
    if t.dt.tz is None:
        t = pd.to_datetime(df[tcol], errors="coerce").dt.tz_localize("UTC")
    # “jst” などが列名にあれば JST 扱い
    if "jst" in tcol.lower(): t = t.dt.tz_convert(JP)
    out = pd.DataFrame({"time": t.dt.tz_convert(JP)})
    out["value"] = pd.to_numeric(df[vcol], errors="coerce") if vcol else pd.to_numeric(df.iloc[:,1], errors="coerce")
    out = out.dropna(subset=["time","value"]).sort_values("time").reset_index(drop=True)
    return out

def resample(df, rule="1min"):
    if df.empty: return df
    g = df.set_index("time").sort_index()[["value"]].resample(rule).mean()
    g["value"] = g["value"].interpolate(limit_direction="both")
    return g.reset_index()

def pick_window(df, key):
    if df.empty: return df
    now_jst = pd.Timestamp.now(tz=JP)
    today = now_jst.normalize()
    if key in ("astra4","rbank9"):
        s = pd.Timestamp(f"{today.date()} 09:00", tz=JP)
        e = pd.Timestamp(f"{today.date()} 15:30", tz=JP)
        w = df[(df["time"]>=s)&(df["time"]<=e)]
    elif key=="ain10":
        tny = df["time"].dt.tz_convert(NY)
        day = pd.Timestamp.now(tz=NY).normalize()
        s = pd.Timestamp(f"{day.date()} 09:30", tz=NY)
        e = pd.Timestamp(f"{day.date()} 16:00", tz=NY)
        w = df[(tny>=s)&(tny<=e)]
    else:  # scoin_plus
        w = df[df["time"] >= (pd.Timestamp.now(tz=JP) - pd.Timedelta(hours=24))]
    if w.empty: w = df.tail(600)
    return w.reset_index(drop=True)

def decide_pct(series_vals):
    """系列の性質から 騰落率(%) を賢く決める"""
    s = pd.to_numeric(series_vals, errors="coerce").dropna()
    if len(s) < 2: return None
    vmin, vmax = float(s.min()), float(s.max())
    vabs_med = float(s.abs().median())
    base, last = float(s.iloc[0]), float(s.iloc[-1])

    CAP = 120.0

    def clip(p):
        if p is None: return None
        return max(-CAP, min(CAP, p))

    # ① 微小レンジ（±0.5以内が多い）⇒ リターン積み上げ（∏(1+v)-1）
    if (vmax - vmin) <= 1.0 and vabs_med <= 0.5:
        prod = 1.0
        for v in s.values: prod *= (1.0 + float(v))
        return clip((prod - 1.0) * 100.0)

    # ② 基準が十分に離れ、かつ符号が一致 ⇒ 比
    if abs(base) > 1e-6 and (base * last) > 0:
        return clip(((last / base) - 1.0) * 100.0)

    # ③ それ以外（符号またぎ/基準小さい）⇒ 差分（%ポイント近似）
    return clip((last - base) * 100.0)

def decorate(ax, title, xl, yl):
    ax.set_title(title, color=TITLE, fontsize=20, pad=12)
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(True, alpha=GRID_A)
    for s in ax.spines.values(): s.set_color(FG)

def save(fig, path): fig.savefig(path, facecolor=BG, bbox_inches="tight"); plt.close(fig)

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    key  = os.environ.get("INDEX_KEY","index").strip().lower()
    name = key.upper().replace("_","")

    intraday_csv = os.path.join(OUTDIR, f"{key}_intraday.csv")
    history_csv  = os.path.join(OUTDIR, f"{key}_history.csv")

    try:
        i = read_intraday(intraday_csv)
        i = pick_window(i, key)
        i = resample(i, "1min")
    except Exception as e:
        print("intraday load fail:", e)
        i = pd.DataFrame(columns=["time","value"])

    delta = decide_pct(i["value"]) if not i.empty else None
    color = UP if (delta is None or delta >= 0) else DOWN

    # 1d
    fig, ax = plt.subplots(figsize=(16,7), layout="constrained")
    decorate(ax, f"{name} (1d)", "Time", "Index Value")
    if not i.empty: ax.plot(i["time"], i["value"], lw=2.2, color=color)
    else: ax.text(0.5,0.5,"No data", transform=ax.transAxes, ha="center", va="center", alpha=0.6)
    save(fig, os.path.join(OUTDIR, f"{key}_1d.png"))

    # 7d / 1m / 1y
    if os.path.exists(history_csv):
        h = pd.read_csv(history_csv)
        if "date" in h and "value" in h:
            h["date"]  = pd.to_datetime(h["date"], errors="coerce")
            h["value"] = pd.to_numeric(h["value"], errors="coerce")
            for days, label in [(7,"7d"),(30,"1m"),(365,"1y")]:
                fig, ax = plt.subplots(figsize=(16,7), layout="constrained")
                decorate(ax, f"{name} ({label})", "Date", "Index Value")
                hh = h.tail(days)
                if len(hh) >= 2:
                    col = UP if hh["value"].iloc[-1] >= hh["value"].iloc[0] else DOWN
                    ax.plot(hh["date"], hh["value"], lw=2.0, color=col)
                else:
                    ax.text(0.5,0.5,"No data", transform=ax.transAxes, ha="center", va="center", alpha=0.5)
                save(fig, os.path.join(OUTDIR, f"{key}_{label}.png"))

    # サイト用の % テキスト
    txt = f"{name} 1d: {(0.0 if delta is None else delta):+0.2f}%"
    with open(os.path.join(OUTDIR, f"{key}_post_intraday.txt"), "w", encoding="utf-8") as f:
        f.write(txt)

if __name__ == "__main__":
    main()
