# scripts/long_charts.py
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

# 1分ごとの安全幅（％ポイント）、セッション上限（％）
PER_MIN_CLIP = 2.0
LIMITS = {"scoin_plus": 30.0, "ain10": 15.0, "astra4": 15.0, "rbank9": 15.0}

# ---------- util ----------
def _lower(df): df.columns=[str(c).strip().lower() for c in df.columns]; return df

def _find_time_col(cols):
    for c in cols:
        if re.search(r"time|日時|date|datetime|timestamp|時刻", c): return c
    return cols[0] if cols else None

def read_intraday(path:str)->pd.DataFrame:
    """intraday CSV -> DataFrame(time[JST tz-aware], value[float])"""
    if not os.path.exists(path): return pd.DataFrame(columns=["time","value"])
    raw = pd.read_csv(path, dtype=str)
    if raw.empty: return pd.DataFrame(columns=["time","value"])
    df = _lower(raw)
    tcol = _find_time_col(df.columns)
    if tcol is None: return pd.DataFrame(columns=["time","value"])

    # 値カラム推定
    vcol = None
    for c in df.columns:
        if c == tcol: continue
        if any(k in c for k in ["value","index","score","mean"]):
            vcol = c; break
    if vcol is None:
        for c in df.columns:
            if c == tcol: continue
            try:
                pd.to_numeric(df[c])
                vcol = c; break
            except Exception:
                pass

    # 時刻 -> tz-aware JST
    t = pd.to_datetime(df[tcol], errors="coerce", utc=True)
    if t.dt.tz is None:
        t = pd.to_datetime(df[tcol], errors="coerce").dt.tz_localize("UTC")
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
    """当日セッション（JST / NY）/ クリプトは直近24h"""
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
    else:  # scoin_plus -> rolling 24h
        w = df[df["time"] >= (pd.Timestamp.now(tz=JP) - pd.Timedelta(hours=24))]

    if w.empty:
        w = df.tail(600)
    return w.reset_index(drop=True)

# ---------- delta ----------
def _is_return_series(s: pd.Series) -> bool:
    """分足リターン（％ポイント）っぽいか？"""
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) < 3: return False
    med = float(s.abs().median())     # 典型値
    mx  = float(s.abs().max())
    # 分足の増減（％ポイント）が±数％に収まるレンジなら return 系列とみなす
    return (med <= 2.0 and mx <= 6.0)

def _delta_from_returns(points_pct: pd.Series, cap_session: float) -> float:
    """分足の％ポイント系列 → 積み上げ（各分を±2%にクリップ）"""
    r = pd.to_numeric(points_pct, errors="coerce").dropna()
    if r.empty: return None
    r = r.clip(-PER_MIN_CLIP, PER_MIN_CLIP)  # スパイク除去
    cum = (1.0 + (r / 100.0)).prod() - 1.0
    pct = float(cum * 100.0)
    # セッション上限でクランプ
    return max(-cap_session, min(cap_session, pct))

def _delta_from_levels(values: pd.Series, cap_session: float) -> float:
    """レベル系列 → (last/first - 1) * 100"""
    v = pd.to_numeric(values, errors="coerce").dropna()
    if len(v) < 2: return None
    first, last = float(v.iloc[0]), float(v.iloc[-1])
    if abs(first) < 1e-12: return None
    pct = (last / first - 1.0) * 100.0
    return max(-cap_session, min(cap_session, float(pct)))

def decide_delta(key: str, i_df: pd.DataFrame, h_path: str) -> float:
    """安全運転の騰落率[%]算出。fallback あり。必ず数値を返す。"""
    cap = LIMITS.get(key, 15.0)

    delta = None
    if not i_df.empty:
        s = i_df["value"].astype(float)
        if _is_return_series(s):
            delta = _delta_from_returns(s, cap)
        else:
            delta = _delta_from_levels(s, cap)

    # 異常・欠損なら history へフォールバック（直近2点）
    if delta is None or not pd.notna(delta):
        if os.path.exists(h_path):
            try:
                h = pd.read_csv(h_path)
                h = _lower(h)
                if "date" in h and "value" in h and len(h) >= 2:
                    a = float(pd.to_numeric(h["value"].iloc[-2], errors="coerce"))
                    b = float(pd.to_numeric(h["value"].iloc[-1], errors="coerce"))
                    if abs(a) > 1e-12:
                        delta = max(-cap, min(cap, (b/a - 1.0) * 100.0))
            except Exception:
                pass

    # それでもダメなら 0.00%
    if delta is None or not pd.notna(delta):
        delta = 0.0
    return float(delta)

# ---------- drawing ----------
def decorate(ax, title, xl, yl):
    ax.set_title(title, color=TITLE, fontsize=20, pad=12)
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(True, alpha=GRID_A)
    for s in ax.spines.values(): s.set_color(FG)

def save(fig, path): fig.savefig(path, facecolor=BG, bbox_inches="tight"); plt.close(fig)

# ---------- main ----------
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    key  = os.environ.get("INDEX_KEY","index").strip().lower()
    name = key.upper().replace("_","")

    intraday_csv = os.path.join(OUTDIR, f"{key}_intraday.csv")
    history_csv  = os.path.join(OUTDIR, f"{key}_history.csv")

    # intraday 読み込み → セッション抽出 → 1分リサンプル
    try:
        i = read_intraday(intraday_csv)
        i = pick_window(i, key)
        i = resample(i, "1min")
    except Exception as e:
        print("intraday load fail:", e)
        i = pd.DataFrame(columns=["time","value"])

    # 騰落率 [%]（安全運転）
    delta = decide_delta(key, i, history_csv)

    # 色
    color = UP if delta >= 0 else DOWN

    # 1d 図
    fig, ax = plt.subplots(figsize=(16,7), layout="constrained")
    decorate(ax, f"{name} (1d)", "Time", "Index Value")
    if not i.empty:
        ax.plot(i["time"], i["value"], lw=2.2, color=color)
    else:
        ax.text(0.5,0.5,"No data", transform=ax.transAxes, ha="center", va="center", alpha=0.6)
    save(fig, os.path.join(OUTDIR, f"{key}_1d.png"))

    # 7d / 1m / 1y 図（あれば）
    if os.path.exists(history_csv):
        try:
            h = pd.read_csv(history_csv)
            h = _lower(h)
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
        except Exception as e:
            print("history plotting fail:", e)

    # サイト用テキスト（常に数値を出力）
    txt = f"{name} 1d: {delta:+0.2f}%"
    with open(os.path.join(OUTDIR, f"{key}_post_intraday.txt"), "w", encoding="utf-8") as f:
        f.write(txt)

if __name__ == "__main__":
    main()
