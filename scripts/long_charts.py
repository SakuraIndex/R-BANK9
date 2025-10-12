# scripts/long_charts.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import pandas as pd
import pytz
import matplotlib
import matplotlib.pyplot as plt

JP = pytz.timezone("Asia/Tokyo")
NY = pytz.timezone("America/New_York")

# --- theme ---
BG, FG, TITLE = "#0E1117", "#E6E6E6", "#f2b6c6"
UP, DOWN, GRID_A = "#3bd6c6", "#ff6b6b", 0.25
matplotlib.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": FG,
    "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
    "text.color": FG, "grid.color": FG, "savefig.facecolor": BG,
})

OUTDIR = os.path.join("docs", "outputs")

# ---------- common utils ----------
def _lower(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().lower() for c in df.columns]; return df

def _time_col(cols):
    for k in ("time","timestamp","date","datetime","日時"):
        if k in cols: return k
    for c in cols:
        if c.startswith("unnamed") and ": 0" in c: return c
    for c in cols:
        if ("time" in c) or ("date" in c): return c
    return None

def read_intraday(path: str) -> pd.DataFrame:
    if not os.path.exists(path): return pd.DataFrame(columns=["time","value","volume"])
    raw = pd.read_csv(path, dtype=str)
    if raw.empty: return pd.DataFrame(columns=["time","value","volume"])
    df = _lower(raw.copy())

    # drop comment cols
    drop = [c for c in df.columns if str(c).strip().startswith("#")]
    if drop: df = df.drop(columns=drop)

    tcol = _time_col(df.columns.tolist())
    if tcol is None: return pd.DataFrame(columns=["time","value","volume"])

    # time -> tz-aware JST
    t = pd.to_datetime(df[tcol], errors="coerce", utc=True)
    if t.dt.tz is None:
        t = pd.to_datetime(df[tcol], errors="coerce").dt.tz_localize(JP).dt.tz_convert("UTC")
    t = t.dt.tz_convert(JP)

    out = pd.DataFrame({"time": t})

    # value column or equal-weight mean
    vcol = None
    for c in df.columns:
        if c == tcol: continue
        if any(k in c for k in ("value","index","score","mean")): vcol = c; break
    if vcol:
        out["value"] = pd.to_numeric(df[vcol], errors="coerce")
    else:
        num = []
        for c in df.columns:
            if c == tcol: continue
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum(): num.append(c)
        if not num: return pd.DataFrame(columns=["time","value","volume"])
        out["value"] = df[num].apply(lambda s: pd.to_numeric(s, errors="coerce")).mean(axis=1)

    out["volume"] = 0
    return out.dropna(subset=["time","value"]).sort_values("time").reset_index(drop=True)

def resample_1min(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    tmp = df.set_index("time").sort_index()
    g = tmp[["value"]].resample("1min").mean()
    g["value"] = g["value"].interpolate(limit_direction="both")
    g["volume"] = 0
    return g.reset_index()

def read_history(path: str) -> pd.DataFrame:
    if not os.path.exists(path): return pd.DataFrame(columns=["date","value"])
    df = _lower(pd.read_csv(path))
    if "date" not in df.columns or "value" not in df.columns:
        return pd.DataFrame(columns=["date","value"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["date","value"]).sort_values("date").reset_index(drop=True)

def session_window(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Pick proper window per index."""
    if df.empty: return df

    now_jst = pd.Timestamp.now(tz=JP)
    today = now_jst.normalize()

    if key in ("astra4","rbank9"):
        s = pd.Timestamp(f"{today.date()} 09:00:00", tz=JP)
        e = pd.Timestamp(f"{today.date()} 15:30:00", tz=JP)
        w = df[(df["time"]>=s)&(df["time"]<=e)]
        if w.empty:  # fallback: previous session
            y = today - pd.Timedelta(days=1)
            s = pd.Timestamp(f"{y.date()} 09:00:00", tz=JP)
            e = pd.Timestamp(f"{y.date()} 15:30:00", tz=JP)
            w = df[(df["time"]>=s)&(df["time"]<=e)]
        return w.reset_index(drop=True)

    if key == "ain10":
        # convert timestamps to NY
        t_ny = df["time"].dt.tz_convert(NY)
        day = pd.Timestamp.now(tz=NY).normalize()
        s = pd.Timestamp(f"{day.date()} 09:30:00", tz=NY)
        e = pd.Timestamp(f"{day.date()} 16:00:00", tz=NY)
        m = (t_ny>=s)&(t_ny<=e)
        w = df[m]
        if w.empty:
            y = day - pd.Timedelta(days=1)
            s = pd.Timestamp(f"{y.date()} 09:30:00", tz=NY)
            e = pd.Timestamp(f"{y.date()} 16:00:00", tz=NY)
            m = (t_ny>=s)&(t_ny<=e)
            w = df[m]
        return w.reset_index(drop=True)

    if key == "scoin_plus":
        # last 24h rolling window (24/7)
        from_ts = pd.Timestamp.now(tz=JP) - pd.Timedelta(hours=24)
        w = df[df["time"]>=from_ts]
        # as ultimate fallback, take last 600 points
        if w.empty: w = df.tail(600)
        return w.reset_index(drop=True)

    return df

def robust_delta_pct(values: pd.Series) -> float|None:
    """Use cumulative return vs first/last; pick sensible one."""
    vals = pd.to_numeric(values, errors="coerce").dropna().values
    if len(vals) < 2: return None
    MIN_DEN = 1e-6

    # cumulative product (treat big absolute ticks as % points)
    prod = 1.0
    for v in vals:
        inc = v/100.0 if abs(v) > 0.5 else v
        prod *= (1.0 + inc)
    p_mul = (prod - 1.0) * 100.0

    # first/last ratio
    first = vals[0]
    p_lvl = None if abs(first) < MIN_DEN else ((vals[-1]/first) - 1.0) * 100.0

    cand = [x for x in (p_mul, p_lvl) if x is not None and abs(x) <= 100.0]
    if not cand: return 0.0
    return min(cand, key=lambda x: abs(x))

def decorate(ax, title, xl, yl):
    ax.set_title(title, color=TITLE, fontsize=20, pad=12)
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(True, alpha=GRID_A)
    for sp in ax.spines.values(): sp.set_color(FG)

def save(fig, path): fig.savefig(path, facecolor=BG, bbox_inches="tight"); plt.close(fig)

# ---------- main ----------
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    key = os.environ.get("INDEX_KEY", "index").strip().lower()
    name = key.upper().replace("_","")

    intraday_csv = os.path.join(OUTDIR, f"{key}_intraday.csv")
    history_csv  = os.path.join(OUTDIR, f"{key}_history.csv")

    # 1d
    try:
        i = read_intraday(intraday_csv)
        i = session_window(i, key)
        i = resample_1min(i)
    except Exception as e:
        print(f"[WARN] intraday load failed: {e}")
        i = pd.DataFrame(columns=["time","value","volume"])

    delta = robust_delta_pct(i["value"]) if not i.empty else None
    color = UP if (delta is None or delta >= 0) else DOWN

    fig, ax = plt.subplots(figsize=(16,7), layout="constrained")
    decorate(ax, f"{name} (1d)", "Time", "Index Value")
    if not i.empty:
        ax.plot(i["time"], i["value"], linewidth=2.4, color=color)
    else:
        ax.text(0.5,0.5,"No data", transform=ax.transAxes, ha="center", va="center", alpha=0.6)
    save(fig, os.path.join(OUTDIR, f"{key}_1d.png"))

    # 7d/1m/1y
    h = read_history(history_csv)
    def plot_hist(n, label, out):
        fig, ax = plt.subplots(figsize=(16,7), layout="constrained")
        decorate(ax, f"{name} ({label})", "Date", "Index Value")
        hh = h.tail(n)
        if len(hh) >= 2:
            col = UP if hh["value"].iloc[-1] >= hh["value"].iloc[0] else DOWN
            ax.plot(hh["date"], hh["value"], linewidth=2.2, color=col)
        elif len(hh)==1:
            ax.plot(hh["date"], hh["value"], marker="o", linewidth=0, color=UP)
        else:
            ax.text(0.5,0.5,"No data", transform=ax.transAxes, ha="center", va="center", alpha=0.5)
        save(fig, os.path.join(OUTDIR, out))
    plot_hist(7, "7d", f"{key}_7d.png")
    plot_hist(30,"1m", f"{key}_1m.png")
    plot_hist(365,"1y", f"{key}_1y.png")

    # write % for the site
    txt = f"{name} 1d: {(0.0 if delta is None else delta):+0.2f}%"
    with open(os.path.join(OUTDIR, f"{key}_post_intraday.txt"), "w", encoding="utf-8") as f:
        f.write(txt)

    # stamp
    with open(os.path.join(OUTDIR, "_last_run.txt"), "w") as f:
        f.write(pd.Timestamp.now(tz=JP).isoformat())

if __name__ == "__main__":
    main()
