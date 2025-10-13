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

# 上限（異常値ガード）
CAPS = {"scoin_plus": 50.0, "ain10": 25.0, "astra4": 20.0, "rbank9": 20.0}

# ---------- util ----------
def _lower(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def _find_time_col(cols) -> str | None:
    for c in cols:
        if re.search(r"time|日時|date|datetime|timestamp|時刻", c):
            return c
    return cols[0] if cols else None

def _parse_time(series: pd.Series) -> pd.Series:
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

    # ---- 値列の拾い方を全面見直し ----
    num_cols = []
    for c in df.columns:
        if c == tcol:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() > 0:
            num_cols.append(s)

    if not num_cols:
        return pd.DataFrame(columns=["time", "value"])

    v = pd.concat(num_cols, axis=1).mean(axis=1, skipna=True)

    out = pd.DataFrame({"time": t, "value": v})
    out = out.dropna(subset=["time", "value"]).sort_values("time").reset_index(drop=True)
    return out

def resample_1min(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    g = df.set_index("time").sort_index()[["value"]].resample("1min").mean()
    g["value"] = g["value"].interpolate(limit_direction="both")
    return g.reset_index()

def _session_window_jst(df: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    s = pd.Timestamp(f"{day.date()} 09:00", tz=JP)
    e = pd.Timestamp(f"{day.date()} 15:30", tz=JP)
    return df[(df["time"] >= s) & (df["time"] <= e)]

def _session_window_ny(df: pd.DataFrame, day_ny: pd.Timestamp) -> pd.DataFrame:
    tny = df["time"].dt.tz_convert(NY)
    s = pd.Timestamp(f"{day_ny.date()} 09:30", tz=NY)
    e = pd.Timestamp(f"{day_ny.date()} 16:00", tz=NY)
    return df[(tny >= s) & (tny <= e)]

def pick_window(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """今日のセッション → 空なら前営業日 → それも空なら末尾600"""
    if df.empty:
        return df

    today_jst = pd.Timestamp.now(tz=JP).normalize()

    if key in ("astra4", "rbank9"):
        win = _session_window_jst(df, today_jst)
        if win.empty:
            win = _session_window_jst(df, today_jst - pd.Timedelta(days=1))
    elif key == "ain10":
        today_ny = pd.Timestamp.now(tz=NY).normalize()
        win = _session_window_ny(df, today_ny)
        if win.empty:
            win = _session_window_ny(df, today_ny - pd.Timedelta(days=1))
    else:  # scoin_plus → 直近24h
        win = df[df["time"] >= (pd.Timestamp.now(tz=JP) - pd.Timedelta(hours=24))]
        if win.empty:
            win = df.tail(600)

    if win.empty:
        win = df.tail(600)

    return win.sort_values("time").reset_index(drop=True)

# ---------- % 計算 ----------
def _clip(v: float | None, cap: float) -> float | None:
    if v is None:
        return None
    return max(-cap, min(cap, v))

def _pct_level(s: pd.Series) -> float | None:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) < 2:
        return None
    first, last = float(s.iloc[0]), float(s.iloc[-1])
    if abs(first) < 1e-12:
        return None
    return (last / first - 1.0) * 100.0

def _pct_product(s: pd.Series, assume_percent: bool) -> float | None:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) < 2:
        return None
    prod = 1.0
    for v in s:
        x = float(v)
        prod *= (1 + (x / 100.0 if assume_percent else x))
    return (prod - 1.0) * 100.0

def calc_percent(key: str, values: pd.Series) -> float | None:
    s = pd.to_numeric(values, errors="coerce").dropna()
    if len(s) < 2:
        return None

    cap = CAPS.get(key, 30.0)

    if key == "scoin_plus":
        # 2通り試して小さい方を採用（%表記か倍率か混在に強い）
        p1 = _pct_product(s, assume_percent=True)   # 値が%（±1前後）
        p2 = _pct_product(s, assume_percent=False)  # 値が倍率（±0.01前後）
        cands = [p for p in (p1, p2) if p is not None]
        if not cands:
            # 最後の砦
            p = _pct_level(s)
        else:
            p = min(cands, key=lambda x: abs(x))
    else:
        p = _pct_level(s)

    return _clip(p, cap)

def draw_chart(df: pd.DataFrame, key: str, name: str, delta: float | None) -> None:
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

    df = pd.DataFrame()
    try:
        df = read_intraday(os.path.join(OUTDIR, f"{key}_intraday.csv"))
        df = pick_window(df, key)
        df = resample_1min(df)
    except Exception as e:
        print("intraday load fail:", e)

    delta = calc_percent(key, df["value"]) if not df.empty else None
    draw_chart(df, key, name, delta)

    # サイト側の表示用テキスト（PNGと同じ値を出力）
    with open(os.path.join(OUTDIR, f"{key}_post_intraday.txt"), "w", encoding="utf-8") as f:
        f.write(f"{name} 1d: {(0.0 if delta is None else float(delta)):+0.2f}%")

if __name__ == "__main__":
    main()
