# scripts/long_charts.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import re
import pandas as pd
import pytz
import matplotlib
import matplotlib.pyplot as plt

# ---- TZ ----
JP = pytz.timezone("Asia/Tokyo")
NY = pytz.timezone("America/New_York")

# ---- Theme ----
BG, FG, TITLE = "#0E1117", "#E6E6E6", "#f2b6c6"
UP, DOWN, GRID_A = "#3bd6c6", "#ff6b6b", 0.25

matplotlib.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "axes.edgecolor": FG,
    "axes.labelcolor": FG,
    "xtick.color": FG,
    "ytick.color": FG,
    "text.color": FG,
    "grid.color": FG,
    "savefig.facecolor": BG,
})

OUTDIR = os.path.join("docs", "outputs")

# ===== util =====
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
    # もし tz 情報が無ければ UTC とみなす
    if getattr(t.dt, "tz", None) is None:
        t = pd.to_datetime(series, errors="coerce").dt.tz_localize("UTC")
    # 画面表示・集計は JST に統一
    return t.dt.tz_convert(JP)

def _rowwise_numeric_mean(df_num: pd.DataFrame) -> pd.Series:
    """複数銘柄列があるケース（R-BANK9など）を行平均で 1 系列に畳み込む。"""
    if df_num.shape[1] == 0:
        return pd.Series([], dtype=float)
    return df_num.mean(axis=1, skipna=True)

def read_intraday(path: str, index_key: str) -> pd.DataFrame:
    """
    CSV → DataFrame({"time","value"})
    - time: tz-aware (JST)
    - value: 1 系列（R-BANK9 等は行方向平均）
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=["time", "value"])

    raw = pd.read_csv(path, dtype=str)
    if raw.empty:
        return pd.DataFrame(columns=["time", "value"])

    df = _lower(raw)
    tcol = _find_time_col(df.columns)
    if tcol is None:
        return pd.DataFrame(columns=["time", "value"])

    # 時刻
    t = _parse_time(df[tcol])

    # 数値列（time 以外を全部数値化）→ 行平均
    num_cols = []
    for c in df.columns:
        if c == tcol:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() > 0:
            num_cols.append(s)

    if not num_cols:
        return pd.DataFrame(columns=["time", "value"])

    df_num = pd.concat(num_cols, axis=1)
    v = _rowwise_numeric_mean(df_num)

    out = pd.DataFrame({"time": t, "value": v})
    out = out.dropna(subset=["time", "value"]).sort_values("time").reset_index(drop=True)
    return out

def resample_1min(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    g = df.set_index("time").sort_index()[["value"]].resample("1min").mean()
    g["value"] = g["value"].interpolate(limit_direction="both")
    return g.reset_index()

def pick_window(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """指数ごとの当日ウィンドウ切り出し（fallback: 前営業日 / 直近 N 点）"""
    if df.empty:
        return df
    now_jst = pd.Timestamp.now(tz=JP)
    today = now_jst.normalize()

    if key in ("astra4", "rbank9"):
        s = pd.Timestamp(f"{today.date()} 09:00", tz=JP)
        e = pd.Timestamp(f"{today.date()} 15:30", tz=JP)
        w = df[(df["time"] >= s) & (df["time"] <= e)]
        if w.empty:
            y = today - pd.Timedelta(days=1)
            s2 = pd.Timestamp(f"{y.date()} 09:00", tz=JP)
            e2 = pd.Timestamp(f"{y.date()} 15:30", tz=JP)
            w = df[(df["time"] >= s2) & (df["time"] <= e2)]
    elif key == "ain10":
        tny = df["time"].dt.tz_convert(NY)
        day = pd.Timestamp.now(tz=NY).normalize()
        s = pd.Timestamp(f"{day.date()} 09:30", tz=NY)
        e = pd.Timestamp(f"{day.date()} 16:00", tz=NY)
        w = df[(tny >= s) & (tny <= e)]
        if w.empty:
            y = day - pd.Timedelta(days=1)
            s2 = pd.Timestamp(f"{y.date()} 09:30", tz=NY)
            e2 = pd.Timestamp(f"{y.date()} 16:00", tz=NY)
            w = df[(tny >= s2) & (tny <= e2)]
    else:  # scoin_plus
        # 直近 24h を対象に
        w = df[df["time"] >= (pd.Timestamp.now(tz=JP) - pd.Timedelta(hours=24))]

    if w.empty:
        w = df.tail(600)
    return w.reset_index(drop=True)

# ---- % 計算：指数別の明示ロジック ----
CAPS = {
    "scoin_plus": 50.0,   # 仮想通貨（%ポイント積み上げ）上限
    "ain10": 25.0,        # 米株10銘柄指数
    "astra4": 20.0,
    "rbank9": 20.0,
}

def _clip(x: float | None, cap: float) -> float | None:
    if x is None:
        return None
    if x > cap:
        return cap
    if x < -cap:
        return -cap
    return x

def pct_for_scoin_plus(values: pd.Series) -> float | None:
    """S-COIN+ は 1 分足の“%ポイント”が並んでいる → 複利積み上げ"""
    s = pd.to_numeric(values, errors="coerce").dropna()
    if len(s) < 2:
        return None
    # 値は %ポイント とみなし、/100 して積み上げ
    prod = 1.0
    for v in s.values:
        prod *= (1.0 + (float(v) / 100.0))
    return (prod - 1.0) * 100.0

def pct_for_level(values: pd.Series) -> float | None:
    """水準系列（Astra4 / R-BANK9 / AIN-10）: 始値比"""
    s = pd.to_numeric(values, errors="coerce").dropna()
    if len(s) < 2:
        return None
    first = float(s.iloc[0])
    last = float(s.iloc[-1])
    if abs(first) < 1e-9:
        return None
    return ((last / first) - 1.0) * 100.0

def calc_delta_percent(key: str, values: pd.Series) -> float | None:
    if key == "scoin_plus":
        p = pct_for_scoin_plus(values)
    else:
        p = pct_for_level(values)
    return _clip(p, CAPS.get(key, 30.0))

# ---- draw ----
def decorate(ax, title, xl, yl):
    ax.set_title(title, color=TITLE, fontsize=20, pad=12)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.grid(True, alpha=GRID_A)
    for s in ax.spines.values():
        s.set_color(FG)

def save(fig, path):
    fig.savefig(path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)

# ===== main =====
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    key = os.environ.get("INDEX_KEY", "index").strip().lower()
    name = key.upper().replace("_", "")

    intraday_csv = os.path.join(OUTDIR, f"{key}_intraday.csv")
    history_csv = os.path.join(OUTDIR, f"{key}_history.csv")

    # 1) intraday 読み込み → 切り出し → 1分足化
    try:
        i = read_intraday(intraday_csv, key)
        i = pick_window(i, key)
        i = resample_1min(i)
    except Exception as e:
        print("intraday load fail:", e)
        i = pd.DataFrame(columns=["time", "value"])

    # 2) 騰落率（指数別の“正しい”ロジック）
    delta = calc_delta_percent(key, i["value"]) if not i.empty else None

    # 3) 線色（騰落率と一致）
    line_color = UP if (delta is not None and delta >= 0) else DOWN

    # ---- 1d ----
    fig, ax = plt.subplots(figsize=(16, 7), layout="constrained")
    decorate(ax, f"{name} (1d)", "Time", "Index Value")
    if not i.empty:
        ax.plot(i["time"], i["value"], lw=2.2, color=line_color)
    else:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", alpha=0.6)
    save(fig, os.path.join(OUTDIR, f"{key}_1d.png"))

    # ---- 7d / 1m / 1y ----
    if os.path.exists(history_csv):
        try:
            h = pd.read_csv(history_csv)
            if {"date", "value"}.issubset({c.lower() for c in h.columns}):
                # 列名を素直に参照（大小区別なし対応）
                cols = {c.lower(): c for c in h.columns}
                h["date"] = pd.to_datetime(h[cols["date"]], errors="coerce")
                h["value"] = pd.to_numeric(h[cols["value"]], errors="coerce")

                for days, label in [(7, "7d"), (30, "1m"), (365, "1y")]:
                    fig, ax = plt.subplots(figsize=(16, 7), layout="constrained")
                    decorate(ax, f"{name} ({label})", "Date", "Index Value")
                    hh = h.tail(days)
                    if len(hh) >= 2:
                        col = UP if hh["value"].iloc[-1] >= hh["value"].iloc[0] else DOWN
                        ax.plot(hh["date"], hh["value"], lw=2.0, color=col)
                    else:
                        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                                ha="center", va="center", alpha=0.5)
                    save(fig, os.path.join(OUTDIR, f"{key}_{label}.png"))
        except Exception as e:
            print("history draw fail:", e)

    # 4) サイト用の % テキスト
    txt_val = 0.0 if delta is None else float(delta)
    with open(os.path.join(OUTDIR, f"{key}_post_intraday.txt"), "w", encoding="utf-8") as f:
        f.write(f"{name} 1d: {txt_val:+0.2f}%")

if __name__ == "__main__":
    main()
