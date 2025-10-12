# scripts/long_charts.py
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

# ---------------- utils ----------------
def _lower(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def read_intraday(path: str) -> pd.DataFrame:
    """
    intraday CSV を読み、JST tz-aware の time と value(=等加重平均) を返す。
    - time 列: time/date/datetime/timestamp/日時/時刻 を自動検出。無ければ先頭列
    - 値列: value/index/score/mean を優先。無ければ「時刻以外の数値列の平均」
    - 時刻は tz-aware にして JST に揃える（naive は JST ローカライズ）
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=["time", "value"])

    raw = pd.read_csv(path, dtype=str)
    if raw.empty:
        return pd.DataFrame(columns=["time", "value"])

    df = _lower(raw.copy())

    # time col
    tcol = None
    for c in df.columns:
        if re.search(r"(time|date|datetime|timestamp|日時|時刻)", c, re.I):
            tcol = c; break
    if tcol is None:
        tcol = df.columns[0]

    # numeric candidates
    prefer = [c for c in df.columns if c != tcol and re.search(r"(value|index|score|mean)", c, re.I)]
    numeric_cols = []
    for c in df.columns:
        if c == tcol: continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() > 0:
            numeric_cols.append((c, s))

    if prefer:
        chosen = None
        for c in prefer:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() > 0:
                chosen = s; break
        if chosen is None:
            if not numeric_cols:
                return pd.DataFrame(columns=["time", "value"])
            val = pd.concat([s for _, s in numeric_cols], axis=1).mean(axis=1)
        else:
            val = chosen
    else:
        if not numeric_cols:
            return pd.DataFrame(columns=["time", "value"])
        val = pd.concat([s for _, s in numeric_cols], axis=1).mean(axis=1)

    # to JST tz-aware
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

def resample(df: pd.DataFrame, rule: str = "1min") -> pd.DataFrame:
    if df.empty: return df
    g = df.set_index("time").sort_index()[["value"]].resample(rule).mean()
    g["value"] = g["value"].interpolate(limit_direction="both")
    return g.reset_index()

def pick_window(df: pd.DataFrame, key: str) -> pd.DataFrame:
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
    else:  # scoin_plus は rolling 24h
        w = df[df["time"] >= (pd.Timestamp.now(tz=JP) - pd.Timedelta(hours=24))]
    if w.empty: w = df.tail(600)
    return w.reset_index(drop=True)

# ---------------- robust % change ----------------
def decide_pct(series_vals) -> float | None:
    """
    騰落率(%) を堅牢に算出。
    デフォルトは“レベル系列”として (last - first)*100 を採用。
    例外的に「尺度が十分大きく、初値と終値の符号が同じ」の場合のみ
    比率 ((last/first)-1)*100 を採用。
    """
    s = pd.to_numeric(series_vals, errors="coerce").dropna()
    if len(s) < 2:
        return None

    base, last = float(s.iloc[0]), float(s.iloc[-1])
    rng = float((s.max() - s.min()))
    med = float(s.abs().median())

    CAP = 60.0  # 上下限（暴走抑止）

    def clip(x: float | None) -> float | None:
        if x is None: return None
        return max(-CAP, min(CAP, x))

    # 1) デフォルト（指数レベル）: 差分を%換算
    diff_pct = (last - base) * 100.0

    # 2) 比率を安全に使える条件:
    #    - 初値が十分離れている（0.1 以上、または中央値が 1 以上）
    #    - 初値と終値の符号が一致（ゼロ割や符号またぎを避ける）
    use_ratio = (abs(base) >= 0.1 or med >= 1.0) and (base * last > 0)

    if use_ratio:
        ratio_pct = ((last / base) - 1.0) * 100.0
        # 比率と差分に極端な乖離がある場合は、より現実的な値（小さい方）を採用
        if abs(ratio_pct) > abs(diff_pct) * 3:
            return clip(diff_pct)
        return clip(ratio_pct)

    return clip(diff_pct)

# ---------------- plot helpers ----------------
def decorate(ax, title: str, xl: str, yl: str):
    ax.set_title(title, color=TITLE, fontsize=20, pad=12)
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    ax.grid(True, alpha=GRID_A)
    for sp in ax.spines.values():
        sp.set_color(FG)

def save(fig, path: str):
    fig.savefig(path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)

# ---------------- main ----------------
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    key  = os.environ.get("INDEX_KEY", "index").strip().lower()
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
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", alpha=0.6)
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
                    ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                            ha="center", va="center", alpha=0.5)
                save(fig, os.path.join(OUTDIR, f"{key}_{label}.png"))

    # Web表示用テキスト（1d 騰落率）
    txt = f"{name} 1d: {(0.0 if delta is None else delta):+0.2f}%"
    with open(os.path.join(OUTDIR, f"{key}_post_intraday.txt"), "w", encoding="utf-8") as f:
        f.write(txt)

if __name__ == "__main__":
    main()
