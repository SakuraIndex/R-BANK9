# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re
import pandas as pd
import pytz
import matplotlib
import matplotlib.pyplot as plt

# === Timezones ===
JP = pytz.timezone("Asia/Tokyo")
NY = pytz.timezone("America/New_York")

# === Theme ===
BG, FG, TITLE = "#0E1117", "#E6E6E6", "#f2b6c6"
UP, DOWN, GRID_A = "#3bd6c6", "#ff6b6b", 0.25
matplotlib.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": FG,
    "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
    "text.color": FG, "grid.color": FG, "savefig.facecolor": BG,
})

OUTDIR = os.path.join("docs", "outputs")

# === Utility ===
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
        t = t.dt.tz_localize("UTC")
    return t.dt.tz_convert(JP)

def read_intraday(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["time", "value"])
    df = pd.read_csv(path, dtype=str)
    if df.empty:
        return pd.DataFrame(columns=["time", "value"])

    df = _lower(df)
    tcol = _find_time_col(df.columns)
    if not tcol:
        return pd.DataFrame(columns=["time", "value"])
    t = _parse_time(df[tcol])

    num_cols = []
    for c in df.columns:
        if c == tcol:
            continue
        # 列名・データ内容ともに柔軟に数値化
        s = pd.to_numeric(df[c].str.replace(" ", "").str.replace(",", ""), errors="coerce")
        if s.notna().sum() > 0:
            num_cols.append(s)
    if not num_cols:
        print("⚠️ No numeric columns found in", path)
        return pd.DataFrame(columns=["time", "value"])

    v = pd.concat(num_cols, axis=1).mean(axis=1, skipna=True)
    out = pd.DataFrame({"time": t, "value": v}).dropna().sort_values("time")
    return out.reset_index(drop=True)

def resample_1min(df):
    if df.empty:
        return df
    g = df.set_index("time").sort_index().resample("1min").mean()
    g["value"] = g["value"].interpolate(limit_direction="both")
    return g.reset_index()

def pick_window(df, key):
    if df.empty:
        return df
    now_jst = pd.Timestamp.now(tz=JP)
    today = now_jst.normalize()
    if key in ("astra4", "rbank9"):
        s, e = pd.Timestamp(f"{today.date()} 09:00", tz=JP), pd.Timestamp(f"{today.date()} 15:30", tz=JP)
        w = df[(df["time"] >= s) & (df["time"] <= e)]
        if w.empty:
            y = today - pd.Timedelta(days=1)
            s2, e2 = pd.Timestamp(f"{y.date()} 09:00", tz=JP), pd.Timestamp(f"{y.date()} 15:30", tz=JP)
            w = df[(df["time"] >= s2) & (df["time"] <= e2)]
    elif key == "ain10":
        tny = df["time"].dt.tz_convert(NY)
        day = pd.Timestamp.now(tz=NY).normalize()
        s, e = pd.Timestamp(f"{day.date()} 09:30", tz=NY), pd.Timestamp(f"{day.date()} 16:00", tz=NY)
        w = df[(tny >= s) & (tny <= e)]
        if w.empty:
            y = day - pd.Timedelta(days=1)
            s2, e2 = pd.Timestamp(f"{y.date()} 09:30", tz=NY), pd.Timestamp(f"{y.date()} 16:00", tz=NY)
            w = df[(tny >= s2) & (tny <= e2)]
    else:  # S-COIN+
        w = df[df["time"] >= (pd.Timestamp.now(tz=JP) - pd.Timedelta(hours=24))]
    return (w if not w.empty else df.tail(600)).reset_index(drop=True)

# === % change ===
CAPS = {"scoin_plus": 50.0, "ain10": 25.0, "astra4": 20.0, "rbank9": 20.0}

def _clip(v, cap): return None if v is None else max(-cap, min(cap, v))

def calc_percent(key, s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) < 2:
        return None
    first, last = float(s.iloc[0]), float(s.iloc[-1])
    if abs(first) < 1e-12:
        return None
    # S-COIN+ 特殊処理
    if key == "scoin_plus":
        p1 = (last - first) / abs(first) * 100.0
        # 累積率も計算
        prod = 1.0
        for v in s:
            x = float(v)
            prod *= (1 + (x / 100.0 if abs(x) < 10 else x))
        p2 = (prod - 1) * 100.0
        p = min((p1, p2), key=lambda x: abs(x))
    else:
        p = (last / first - 1.0) * 100.0
    return _clip(p, CAPS.get(key, 30.0))

def draw_chart(df, key, name, delta):
    fig, ax = plt.subplots(figsize=(16, 7), layout="constrained")
    ax.set_title(f"{name} (1d)", color=TITLE, fontsize=20, pad=12)
    ax.set_xlabel("Time")
    ax.set_ylabel("Index Value")
    ax.grid(True, alpha=GRID_A)
    for s in ax.spines.values(): s.set_color(FG)

    # AIN-10などでも「終値 - 始値」で色を決定（騰落率がズレても視覚は一致）
    color = UP if (not df.empty and df["value"].iloc[-1] >= df["value"].iloc[0]) else DOWN
    if not df.empty:
        ax.plot(df["time"], df["value"], lw=2.2, color=color)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    fig.savefig(os.path.join(OUTDIR, f"{key}_1d.png"), facecolor=BG, bbox_inches="tight")
    plt.close(fig)

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    key = os.environ.get("INDEX_KEY", "index").strip().lower()
    name = key.upper().replace("_", "")

    df = read_intraday(os.path.join(OUTDIR, f"{key}_intraday.csv"))
    df = pick_window(df, key)
    df = resample_1min(df)
    delta = calc_percent(key, df["value"]) if not df.empty else None
    draw_chart(df, key, name, delta)
    with open(os.path.join(OUTDIR, f"{key}_post_intraday.txt"), "w", encoding="utf-8") as f:
        f.write(f"{name} 1d: {(0.0 if delta is None else delta):+0.2f}%")

if __name__ == "__main__":
    main()
