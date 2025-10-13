# scripts/long_charts.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re
import pandas as pd
import pytz
import matplotlib
import matplotlib.pyplot as plt

JP = pytz.timezone("Asia/Tokyo")
BG, FG, TITLE = "#0E1117", "#E6E6E6", "#f2b6c6"
UP, DOWN, GRID_A = "#3bd6c6", "#ff6b6b", 0.25
matplotlib.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": FG,
    "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
    "text.color": FG, "grid.color": FG, "savefig.facecolor": BG,
})

OUTDIR = os.path.join("docs", "outputs")
KEY   = "rbank9"
NAME  = "RBANK9"

def _lower(df): df.columns=[str(c).strip() for c in df.columns]; return df
def _find_time_col(cols):
    for c in cols:
        if re.search(r"time|日時|date|datetime|timestamp|時刻", str(c), re.I): return c
    return cols[0] if cols else None

def read_intraday(path:str)->pd.DataFrame:
    """rbank9_intraday.csv → time(JST), value(合成値=等金額加重平均)"""
    if not os.path.exists(path): 
        return pd.DataFrame(columns=["time","value"])
    raw = pd.read_csv(path, dtype=str)
    if raw.empty:
        return pd.DataFrame(columns=["time","value"])
    df = _lower(raw)
    tcol = _find_time_col(df.columns)
    if tcol is None:
        return pd.DataFrame(columns=["time","value"])

    # すでに合成列があるか？
    agg_candidates = [c for c in df.columns if c.strip().upper() in ("R_BANK9","RBANK9","R-BANK9")]
    # 銘柄列（数値列）を拾う
    num_cols = []
    for c in df.columns:
        if c == tcol: continue
        try:
            pd.to_numeric(df[c])
            num_cols.append(c)
        except Exception:
            pass

    # 等金額平均：合成列があればそれを採用、無ければ数値列の平均
    if agg_candidates:
        vseries = pd.to_numeric(df[agg_candidates[0]], errors="coerce")
    else:
        if not num_cols:
            return pd.DataFrame(columns=["time","value"])
        vseries = pd.to_numeric(df[num_cols], errors="coerce").mean(axis=1)

    # time を JST tz-aware へ
    t = pd.to_datetime(df[tcol], errors="coerce", utc=True)
    if t.dt.tz is None:
        t = pd.to_datetime(df[tcol], errors="coerce").dt.tz_localize("UTC")
    t = t.dt.tz_convert(JP)

    out = pd.DataFrame({"time": t, "value": vseries})
    out = out.dropna(subset=["time","value"]).sort_values("time").reset_index(drop=True)
    return out

def pick_window_jst(df: pd.DataFrame) -> pd.DataFrame:
    """東証 09:00-15:30 の当日窓"""
    if df.empty: return df
    now_jst = pd.Timestamp.now(tz=JP)
    today = now_jst.normalize()
    s = pd.Timestamp(f"{today.date()} 09:00", tz=JP)
    e = pd.Timestamp(f"{today.date()} 15:30", tz=JP)
    w = df[(df["time"]>=s)&(df["time"]<=e)]
    return (w if not w.empty else df.tail(600)).reset_index(drop=True)

def calc_delta(series: pd.Series) -> float | None:
    """等加重指数：差分×100（%ポイント近似）"""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 2: return None
    base, last = float(s.iloc[0]), float(s.iloc[-1])
    return (last - base) * 100.0

def decorate(ax, title, xl, yl):
    ax.set_title(title, color=TITLE, fontsize=20, pad=12)
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(True, alpha=GRID_A)
    for sp in ax.spines.values(): sp.set_color(FG)

def save(fig, path):
    fig.savefig(path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    intraday_csv = os.path.join(OUTDIR, f"{KEY}_intraday.csv")
    history_csv  = os.path.join(OUTDIR, f"{KEY}_history.csv")

    try:
        i = read_intraday(intraday_csv)
        i = pick_window_jst(i)
        if not i.empty:
            i = (i.set_index("time")[["value"]]
                   .resample("1min").mean()
                   .interpolate(limit_direction="both")
                   .reset_index())
    except Exception as e:
        print("intraday load fail:", e)
        i = pd.DataFrame(columns=["time","value"])

    delta = calc_delta(i["value"]) if not i.empty else None
    color = UP if (delta is not None and delta >= 0) else DOWN

    fig, ax = plt.subplots(figsize=(16,7), layout="constrained")
    decorate(ax, f"{NAME} (1d)", "Time", "Index value")
    if not i.empty:
        ax.plot(i["time"], i["value"], lw=2.2, color=color)
    else:
        ax.text(0.5,0.5,"No data", transform=ax.transAxes, ha="center", va="center", alpha=0.6)
    save(fig, os.path.join(OUTDIR, f"{KEY}_1d.png"))

    if os.path.exists(history_csv):
        h = pd.read_csv(history_csv)
        if "date" in h and "value" in h:
            h["date"]  = pd.to_datetime(h["date"], errors="coerce")
            h["value"] = pd.to_numeric(h["value"], errors="coerce")
            for days, label in [(7,"7d"),(30,"1m"),(365,"1y")]:
                fig, ax = plt.subplots(figsize=(16,7), layout="constrained")
                decorate(ax, f"{NAME} ({label})", "Date", "Index value")
                hh = h.tail(days)
                if len(hh) >= 2:
                    col = UP if hh["value"].iloc[-1] >= hh["value"].iloc[0] else DOWN
                    ax.plot(hh["date"], hh["value"], lw=2.0, color=col)
                else:
                    ax.text(0.5,0.5,"No data", transform=ax.transAxes, ha="center", va="center", alpha=0.5)
                save(fig, os.path.join(OUTDIR, f"{KEY}_{label}.png"))

    txt = f"R-BANK9 1d: {(0.0 if delta is None else delta):+0.2f}%"
    with open(os.path.join(OUTDIR, f"{KEY}_post_intraday.txt"), "w", encoding="utf-8") as f:
        f.write(txt)

if __name__ == "__main__":
    main()
