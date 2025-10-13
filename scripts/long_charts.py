# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re
import pandas as pd
import pytz
import matplotlib
import matplotlib.pyplot as plt

# ---------- TZ ----------
JP = pytz.timezone("Asia/Tokyo")
NY = pytz.timezone("America/New_York")

# ---------- Theme ----------
BG, FG, TITLE = "#0E1117", "#E6E6E6", "#f2b6c6"
UP, DOWN, GRID_A = "#3bd6c6", "#ff6b6b", 0.25

matplotlib.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": FG,
    "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
    "text.color": FG, "grid.color": FG, "savefig.facecolor": BG,
})

OUTDIR = os.path.join("docs", "outputs")

# ---------- util ----------
def _lower(df):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def _find_time_col(cols):
    for c in cols:
        if re.search(r"time|日時|date|datetime|timestamp|時刻", c):
            return c
    return cols[0] if cols else None

def _parse_time(series):
    t = pd.to_datetime(series, errors="coerce", utc=True)
    if getattr(t.dt, "tz", None) is None:
        t = pd.to_datetime(series, errors="coerce").dt.tz_localize("UTC")
    return t.dt.tz_convert(JP)

def read_intraday(path: str) -> pd.DataFrame:
    """CSV → DataFrame(time[JST], value[float])"""
    if not os.path.exists(path):
        return pd.DataFrame(columns=["time", "value"])

    raw = pd.read_csv(path, dtype=str)
    if raw.empty:
        return pd.DataFrame(columns=["time", "value"])

    df = _lower(raw)
    tcol = _find_time_col(df.columns)
    if tcol is None:
        return pd.DataFrame(columns=["time", "value"])

    t = _parse_time(df[tcol])

    # 数値列（time 以外で有効なもの全部）
    num_cols = []
    for c in df.columns:
        if c == tcol:
            continue
        # 数値列名でも通す
        if re.match(r"^[a-z0-9_.\-]+$", c):
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() > 0:
                num_cols.append(s)

    if not num_cols:
        return pd.DataFrame(columns=["time", "value"])

    v = pd.concat(num_cols, axis=1).mean(axis=1, skipna=True)
    out = pd.DataFrame({"time": t, "value": v})
    out = out.dropna(subset=["time", "value"]).sort_values("time").reset_index(drop=True)
    return out

def resample_1min(df):
    if df.empty:
        return df
    g = df.set_index("time").sort_index()[["value"]].resample("1min").mean()
    g["value"] = g["value"].interpolate(limit_direction="both")
    return g.reset_index()

def pick_window(df, key):
    if df.empty:
        return df
    now_jst = pd.Timestamp.now(tz=JP)
    today = now_jst.normalize()

    if key in ("astra4", "rbank9"):
        s = pd.Timestamp(f"{today.date()} 09:00", tz=JP)
        e = pd.Timestamp(f"{today.date()} 15:30", tz=JP)
        w = df[(df["time"] >= s) & (df["time"] <= e)]
    elif key == "ain10":
        tny = df["time"].dt.tz_convert(NY)
        day = pd.Timestamp.now(tz=NY).normalize()
        s = pd.Timestamp(f"{day.date()} 09:30", tz=NY)
        e = pd.Timestamp(f"{day.date()} 16:00", tz=NY)
        w = df[(tny >= s) & (tny <= e)]
    else:  # scoin_plus
        w = df[df["time"] >= (pd.Timestamp.now(tz=JP) - pd.Timedelta(hours=24))]

    if w.empty:
        w = df.tail(600)
    return w.sort_values("time").reset_index(drop=True)

# ---------- % 計算 ----------
CAPS = {"scoin_plus": 50.0, "ain10": 25.0, "astra4": 20.0, "rbank9": 20.0}

def _clip(v, cap):
    if v is None:
        return None
    return max(-cap, min(cap, v))

def calc_percent(key, values):
    s = pd.to_numeric(values, errors="coerce").dropna()
    if len(s) < 2:
        return None
    s = s.reset_index(drop=True)
    first, last = float(s.iloc[0]), float(s.iloc[-1])

    if key == "scoin_plus":
        # 値が ±10超 → すでに % ではなく倍率系とみなす
        if abs(s).mean() > 10:
            prod = 1.0
            for v in s:
                prod *= (1 + float(v))
            pct = (prod - 1) * 100.0
        else:
            prod = 1.0
            for v in s:
                prod *= (1 + float(v) / 100.0)
            pct = (prod - 1) * 100.0
    else:
        if abs(first) < 1e-9:
            return None
        pct = ((last / first) - 1.0) * 100.0

    return _clip(pct, CAPS.get(key, 30.0))

def draw_chart(df, key, name, delta):
    fig, ax = plt.subplots(figsize=(16, 7), layout="constrained")
    ax.set_title(f"{name} (1d)", color=TITLE, fontsize=20, pad=12)
    ax.set_xlabel("Time")
    ax.set_ylabel("Index Value")
    ax.grid(True, alpha=GRID_A)
    for s in ax.spines.values():
        s.set_color(FG)
    color = UP if (delta is not None and delta >= 0) else DOWN
    if not df.empty:
        ax.plot(df["time"], df["value"], lw=2.2, color=color)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    path = os.path.join(OUTDIR, f"{key}_1d.png")
    fig.savefig(path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)

# ---------- main ----------
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    key = os.environ.get("INDEX_KEY", "index").strip().lower()
    name = key.upper().replace("_", "")

    i = pd.DataFrame()
    try:
        i = read_intraday(os.path.join(OUTDIR, f"{key}_intraday.csv"))
        i = pick_window(i, key)
        i = resample_1min(i)
    except Exception as e:
        print("load fail:", e)

    delta = calc_percent(key, i["value"]) if not i.empty else None
    draw_chart(i, key, name, delta)

    txt = 0.0 if delta is None else float(delta)
    with open(os.path.join(OUTDIR, f"{key}_post_intraday.txt"), "w", encoding="utf-8") as f:
        f.write(f"{name} 1d: {txt:+0.2f}%")

if __name__ == "__main__":
    main()
