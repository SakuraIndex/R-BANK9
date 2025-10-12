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

def _lower(df): 
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def read_intraday(path: str) -> pd.DataFrame:
    """長短どちらのCSVにも対応。広い（ワイド）形式は数値列の平均を取る。"""
    if not os.path.exists(path):
        return pd.DataFrame(columns=["time", "value"])
    raw = pd.read_csv(path, dtype=str)
    if raw.empty:
        return pd.DataFrame(columns=["time", "value"])
    df = _lower(raw.copy())

    # 時刻列
    tcol = None
    for c in df.columns:
        if re.search(r"(time|date|datetime|timestamp|日時|時刻)", c, re.I):
            tcol = c; break
    if tcol is None:
        tcol = df.columns[0]

    # 値列（優先候補 → 数値列平均）
    prefer = [c for c in df.columns if c != tcol and re.search(r"(value|index|score|mean)", c, re.I)]
    nums = []
    for c in df.columns:
        if c == tcol: continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() > 0:
            nums.append(s)
    if prefer:
        val = None
        for c in prefer:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() > 0:
                val = s; break
        if val is None and nums:
            val = pd.concat(nums, axis=1).mean(axis=1)
    else:
        val = pd.concat(nums, axis=1).mean(axis=1) if nums else None
    if val is None:
        return pd.DataFrame(columns=["time","value"])

    # 時刻 → JST
    t = pd.to_datetime(df[tcol], errors="coerce", utc=False)
    try:
        if getattr(t.dt, "tz", None) is not None:
            t = t.dt.tz_convert(JP)
        else:
            t = t.dt.tz_localize(JP)
    except Exception:
        t = pd.to_datetime(df[tcol], errors="coerce").dt.tz_localize(JP)

    out = pd.DataFrame({"time": t, "value": pd.to_numeric(val, errors="coerce")})
    return out.dropna(subset=["time","value"]).sort_values("time").reset_index(drop=True)

def resample(df, rule="1min"):
    if df.empty: return df
    g = df.set_index("time").sort_index()[["value"]].resample(rule).mean()
    g["value"] = g["value"].interpolate(limit_direction="both")
    return g.reset_index()

def pick_window(df, key):
    if df.empty: return df
    now_jst = pd.Timestamp.now(tz=JP)
    today = now_jst.normalize()
    if key in ("astra4", "rbank9"):
        s = pd.Timestamp(f"{today.date()} 09:00", tz=JP)
        e = pd.Timestamp(f"{today.date()} 15:30", tz=JP)
        w = df[(df["time"]>=s) & (df["time"]<=e)]
    elif key == "ain10":
        tny = df["time"].dt.tz_convert(NY)
        day = pd.Timestamp.now(tz=NY).normalize()
        s = pd.Timestamp(f"{day.date()} 09:30", tz=NY)
        e = pd.Timestamp(f"{day.date()} 16:00", tz=NY)
        w = df[(tny>=s) & (tny<=e)]
    else:  # scoin_plus
        w = df[df["time"] >= (pd.Timestamp.now(tz=JP) - pd.Timedelta(hours=24))]
    if w.empty: w = df.tail(600)
    return w.reset_index(drop=True)

# ---- 騰落率（自動スケール判定、上限クリップを撤廃） ----
def decide_pct(series_vals):
    """
    値のスケールを見て自動的に計算方法を切替える：
      A) 値が ±0.1 未満の領域で推移 → 変化量を%ポイント扱い: (last - base) * 100
      B) 値の中央値が 5〜500 → レベル系列: (last/base - 1)*100
      C) 上記以外 → 変化量%近似: (last - base) * 100
    """
    s = pd.to_numeric(series_vals, errors="coerce").dropna()
    if len(s) < 2: return None
    base, last = float(s.iloc[0]), float(s.iloc[-1])
    med  = float(s.abs().median())
    rng  = float((s.max() - s.min()))

    # A) “ゼロ近傍”で動く正規化系列（-0.03 など）が該当
    if abs(base) < 0.1 and abs(last) < 0.1 and rng < 0.5:
        return (last - base) * 100.0

    # B) 一般的な指数レベル
    if 5.0 <= med <= 500.0 and abs(base) > 1e-9 and base * last > 0:
        return ((last / base) - 1.0) * 100.0

    # C) それ以外は差分%近似（極端な暴走を避けるためスケール依存）
    return (last - base) * 100.0

def decorate(ax, title, xl, yl):
    ax.set_title(title, color=TITLE, fontsize=20, pad=12)
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    ax.grid(True, alpha=GRID_A)
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
    if not i.empty:
        ax.plot(i["time"], i["value"], lw=2.2, color=color)
    else:
        ax.text(0.5,0.5,"No data", transform=ax.transAxes, ha="center", va="center", alpha=0.6)
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

    # サイトで使うテキスト
    txt = f"{name} 1d: {(0.0 if delta is None else delta):+0.2f}%"
    with open(os.path.join(OUTDIR, f"{key}_post_intraday.txt"), "w", encoding="utf-8") as f:
        f.write(txt)

if __name__ == "__main__":
    main()
