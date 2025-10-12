# scripts/long_charts.py
# -*- coding: utf-8 -*-
"""
Generate long-term charts (1d / 7d / 1m / 1y) for INDEX_KEY.
Also writes docs/outputs/<index>_post_intraday.txt with 1d change (%).
"""

from __future__ import annotations
import os
import pandas as pd
import pytz
import matplotlib
import matplotlib.pyplot as plt

JP_TZ = pytz.timezone("Asia/Tokyo")
SESSION_START = "09:00"
SESSION_END   = "15:30"

BG    = "#0E1117"
FG    = "#E6E6E6"
TITLE = "#f2b6c6"
GRID_A = 0.25
UP  = "#3bd6c6"
DOWN= "#ff6b6b"

matplotlib.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": FG,
    "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
    "text.color": FG, "grid.color": FG, "savefig.facecolor": BG,
})

OUTPUTS_DIR = os.path.join("docs", "outputs")

def _lower(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def _pick_time_col(cols):
    for k in ("time","timestamp","date","datetime","日時"):
        if k in cols: return k
    for c in cols:
        if c.startswith("unnamed") and ": 0" in c: return c
    for c in cols:
        if ("time" in c) or ("date" in c): return c
    return None

def read_any_intraday(path: str) -> pd.DataFrame:
    if not os.path.exists(path): return pd.DataFrame(columns=["time","value","volume"])
    raw = pd.read_csv(path, dtype=str)
    if raw.empty: return pd.DataFrame(columns=["time","value","volume"])
    df = _lower(raw.copy())
    drop = [c for c in df.columns if str(c).strip().startswith("#")]
    if drop: df = df.drop(columns=drop)
    tcol = _pick_time_col(df.columns.tolist())
    if tcol is None: raise KeyError(f"No time-like column in {path}")

    vcol = None
    for c in df.columns:
        if c == tcol: continue
        if any(k in c for k in ("value","index","score","mean")):
            vcol = c; break
    # time to JST tz-aware
    t = pd.to_datetime(df[tcol], errors="coerce", utc=True)
    if t.dt.tz is None:
        t = pd.to_datetime(df[tcol], errors="coerce").dt.tz_localize(JP_TZ).dt.tz_convert("UTC")
    t = t.dt.tz_convert(JP_TZ)
    out = pd.DataFrame({"time": t})
    if vcol:
        out["value"] = pd.to_numeric(df[vcol], errors="coerce")
    else:
        # equal-weight mean for wide table
        vals = df.drop(columns=[tcol]).apply(lambda s: pd.to_numeric(s, errors="coerce"))
        out["value"] = vals.mean(axis=1)
    out["volume"] = 0
    return out.dropna(subset=["time","value"]).sort_values("time").reset_index(drop=True)

def clamp_today_session_jst(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    today = pd.Timestamp.now(tz=JP_TZ).normalize()
    s = pd.Timestamp(f"{today.date()} {SESSION_START}", tz=JP_TZ)
    e = pd.Timestamp(f"{today.date()} {SESSION_END}",   tz=JP_TZ)
    m = (df["time"]>=s)&(df["time"]<=e)
    out = df.loc[m]
    if out.empty:  # 昨日のセッションを表示（データが無い日の予防）
        y = today - pd.Timedelta(days=1)
        s2 = pd.Timestamp(f"{y.date()} {SESSION_START}", tz=JP_TZ)
        e2 = pd.Timestamp(f"{y.date()} {SESSION_END}",   tz=JP_TZ)
        out = df.loc[(df["time"]>=s2)&(df["time"]<=e2)]
    return out.reset_index(drop=True)

def resample_minutes(df: pd.DataFrame, rule: str="1min") -> pd.DataFrame:
    if df.empty: return df
    tmp = df.set_index("time").sort_index()
    out = tmp[["value"]].resample(rule).mean()
    out["value"] = out["value"].interpolate(limit_direction="both")
    out["volume"] = 0
    return out.reset_index()

def read_history(path: str) -> pd.DataFrame:
    if not os.path.exists(path): return pd.DataFrame(columns=["date","value"])
    df = _lower(pd.read_csv(path))
    if "date" not in df.columns or "value" not in df.columns: return pd.DataFrame(columns=["date","value"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["date","value"]).sort_values("date").reset_index(drop=True)

def _decorate(ax, title, xl, yl):
    ax.set_title(title, color=TITLE, fontsize=20, pad=12)
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(True, alpha=GRID_A)
    for sp in ax.spines.values(): sp.set_color(FG)

def _save(fig, path): fig.savefig(path, facecolor=BG, bbox_inches="tight"); plt.close(fig)

def session_delta_pct(values: pd.Series) -> float|None:
    """robust % change within session."""
    vals = pd.to_numeric(values, errors="coerce").dropna().values
    if len(vals) < 2: return None
    MIN_DEN = 1e-6
    # try cumulative product (treat >0.5 as percent points)
    prod = 1.0
    for v in vals:
        inc = v/100.0 if abs(v) > 0.5 else v
        prod *= (1.0 + inc)
    pct_mul = (prod - 1.0) * 100.0
    # try first/last ratio
    first = vals[0]
    pct_lvl = None if abs(first) < MIN_DEN else ((vals[-1]/first) - 1.0) * 100.0
    # choose sensible one
    candidates = [x for x in (pct_mul, pct_lvl) if x is not None and abs(x) <= 50.0]
    if not candidates:  # both extreme → 0%（安全側）
        return 0.0
    # smaller magnitude is usually safer
    return min(candidates, key=lambda x: abs(x))

def main():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    index_key = os.environ.get("INDEX_KEY", "index").strip().lower()
    index_name = index_key.upper().replace("_","")

    intraday_csv = os.path.join(OUTPUTS_DIR, f"{index_key}_intraday.csv")
    history_csv  = os.path.join(OUTPUTS_DIR, f"{index_key}_history.csv")

    # ---- 1d ----
    try:
        i = read_any_intraday(intraday_csv)
        i = clamp_today_session_jst(i)
        i = resample_minutes(i, "1min")
    except Exception as e:
        print(f"[WARN] intraday load failed: {e}")
        i = pd.DataFrame(columns=["time","value","volume"])

    delta_pct = session_delta_pct(i["value"]) if not i.empty else None
    color = UP if (delta_pct is None or delta_pct >= 0) else DOWN

    fig, ax = plt.subplots(figsize=(16,7), layout="constrained")
    _decorate(ax, f"{index_name} (1d)", "Time", "Index Value")
    if not i.empty:
        ax.plot(i["time"], i["value"], linewidth=2.4, color=color)
    else:
        ax.text(0.5,0.5,"No data", transform=ax.transAxes, ha="center", va="center", alpha=0.6)
    _save(fig, os.path.join(OUTPUTS_DIR, f"{index_key}_1d.png"))

    # 7d / 1m / 1y
    h = read_history(history_csv)
    def plot_hist(tail_n: int, label: str, out: str):
        fig, ax = plt.subplots(figsize=(16,7), layout="constrained")
        _decorate(ax, f"{index_name} ({label})", "Date", "Index Value")
        hh = h.tail(tail_n)
        if len(hh) >= 2:
            col = UP if hh["value"].iloc[-1] >= hh["value"].iloc[0] else DOWN
            ax.plot(hh["date"], hh["value"], linewidth=2.2, color=col)
        elif len(hh) == 1:
            ax.plot(hh["date"], hh["value"], marker="o", markersize=6, linewidth=0, color=UP)
            y = hh["value"].iloc[0]; ax.set_ylim(y-0.1, y+0.1)
            ax.text(0.5,0.5,"Only 1 point (need ≥ 2)", transform=ax.transAxes, ha="center", va="center", alpha=0.5)
        else:
            ax.text(0.5,0.5,"No data", transform=ax.transAxes, ha="center", va="center", alpha=0.5)
        _save(fig, os.path.join(OUTPUTS_DIR, out))

    plot_hist(7,   "7d", f"{index_key}_7d.png")
    plot_hist(30,  "1m", f"{index_key}_1m.png")
    plot_hist(365, "1y", f"{index_key}_1y.png")

    # write 1d delta txt for the site
    txt_path = os.path.join(OUTPUTS_DIR, f"{index_key}_post_intraday.txt")
    if delta_pct is None:
        txt = f"{index_name} 1d: 0.00% (no intraday)"
    else:
        txt = f"{index_name} 1d: {delta_pct:+.2f}%"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt)

    # stamp
    with open(os.path.join(OUTPUTS_DIR, "_last_run.txt"), "w") as f:
        f.write(pd.Timestamp.now(tz=JP_TZ).isoformat())

if __name__ == "__main__":
    main()
