# scripts/long_charts.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re
import pandas as pd
import pytz
import matplotlib
import matplotlib.pyplot as plt

# === タイムゾーン設定 ===
JP = pytz.timezone("Asia/Tokyo")
NY = pytz.timezone("America/New_York")

# === デザイン設定 ===
BG, FG, TITLE = "#0E1117", "#E6E6E6", "#f2b6c6"
UP, DOWN, GRID_A = "#3bd6c6", "#ff6b6b", 0.25

matplotlib.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": FG,
    "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
    "text.color": FG, "grid.color": FG, "savefig.facecolor": BG,
})

OUTDIR = os.path.join("docs", "outputs")

# === ヘルパー関数群 ===
def _lower(df): 
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

# === 改良版 read_intraday ===
def read_intraday(path: str) -> pd.DataFrame:
    """
    intraday CSV を読み、JST tz-aware の time と value(=等加重平均) を返す。
    - time 列は「time / date / datetime / timestamp / 日時 など」を自動検出
    - 値列は value/index/score/mean を優先。該当なしでも
      時刻以外の **数値列を全て等加重平均** して value を作る（ワイド形式対応）
    - 時間は「既に tz 付き → JST へ変換」「naive → JST としてローカライズ」
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=["time", "value"])

    raw = pd.read_csv(path, dtype=str)
    if raw.empty:
        return pd.DataFrame(columns=["time", "value"])

    df = _lower(raw.copy())

    # ---- 時刻列の推定 ----
    tcol = None
    for c in df.columns:
        if re.search(r"(time|date|datetime|timestamp|日時|時刻)", str(c), re.I):
            tcol = c
            break
    if tcol is None:
        tcol = df.columns[0]

    # ---- 値列候補の抽出 ----
    prefer = [c for c in df.columns if c != tcol and re.search(r"(value|index|score|mean)", str(c), re.I)]
    numeric_cols = []
    for c in df.columns:
        if c == tcol:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() > 0:
            numeric_cols.append((c, s))

    if prefer:
        chosen = None
        for c in prefer:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() > 0:
                chosen = s
                break
        if chosen is not None:
            val_series = chosen
        else:
            if not numeric_cols:
                return pd.DataFrame(columns=["time", "value"])
            val_series = pd.concat([s for _, s in numeric_cols], axis=1).mean(axis=1)
    else:
        if not numeric_cols:
            return pd.DataFrame(columns=["time", "value"])
        val_series = pd.concat([s for _, s in numeric_cols], axis=1).mean(axis=1)

    # ---- 時刻を JST tz-aware に ----
    t = pd.to_datetime(df[tcol], errors="coerce", utc=False)
    try:
        if getattr(t.dt, "tz", None) is not None:
            t = t.dt.tz_convert(JP)
        else:
            t = t.dt.tz_localize(JP)
    except Exception:
        t = pd.to_datetime(df[tcol], errors="coerce").dt.tz_localize(JP)

    out = pd.DataFrame({"time": t, "value": pd.to_numeric(val_series, errors="coerce")})
    out = out.dropna(subset=["time", "value"]).sort_values("time").reset_index(drop=True)
    return out

# === Resample ===
def resample(df, rule="1min"):
    if df.empty:
        return df
    g = df.set_index("time").sort_index()[["value"]].resample(rule).mean()
    g["value"] = g["value"].interpolate(limit_direction="both")
    return g.reset_index()

# === pick_window ===
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
    return w.reset_index(drop=True)

# === 改良版 decide_pct ===
def decide_pct(series_vals):
    """
    系列の性質から 騰落率(%) を堅牢に決める。
    - 値のスケール（小数/％ポイント）を自動判定して正規化
    - 小さいレンジ or 収益率系列 ⇒ 積み上げ
    - 同符号・基準十分 ⇒ 比率
    - 上記以外 ⇒ 差分（%ポイント近似）
    """
    s_raw = pd.to_numeric(series_vals, errors="coerce").dropna()
    if len(s_raw) < 2:
        return None

    med = float(s_raw.abs().median())
    if 1.0 <= med <= 100.0:
        s = s_raw * 0.01
        scale_type = "percent_points"
    else:
        s = s_raw.copy()
        scale_type = "ratio_like"

    vmin, vmax = float(s.min()), float(s.max())
    vabs_med = float(s.abs().median())
    base, last = float(s.iloc[0]), float(s.iloc[-1])
    CAP = 120.0

    def clip(p):
        if p is None:
            return None
        return max(-CAP, min(CAP, p))

    # ① 微小レンジ → 積み上げ
    if (vmax - vmin) <= 0.01 * 50 and vabs_med <= 0.01 * 25:
        prod = 1.0
        for v in s.values:
            prod *= (1.0 + float(v))
        ret = (prod - 1.0) * 100.0
        print(f"[decide_pct] PRODUCT ({scale_type}) → {ret:.3f}%")
        return clip(ret)

    # ② 同符号 → 比率
    if abs(base) > 1e-9 and (base * last) > 0:
        ret = ((last / base) - 1.0) * 100.0
        print(f"[decide_pct] RATIO ({scale_type}) → {ret:.3f}%")
        return clip(ret)

    # ③ 符号またぎ → 差分
    ret = (last - base) * 100.0
    print(f"[decide_pct] DIFF ({scale_type}) → {ret:.3f}%")
    return clip(ret)

# === decorate & save ===
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

# === main ===
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    key = os.environ.get("INDEX_KEY", "index").strip().lower()
    name = key.upper().replace("_", "")

    intraday_csv = os.path.join(OUTDIR, f"{key}_intraday.csv")
    history_csv = os.path.join(OUTDIR, f"{key}_history.csv")

    try:
        i = read_intraday(intraday_csv)
        i = pick_window(i, key)
        i = resample(i, "1min")
    except Exception as e:
        print("intraday load fail:", e)
        i = pd.DataFrame(columns=["time", "value"])

    delta = decide_pct(i["value"]) if not i.empty else None
    color = UP if (delta is None or delta >= 0) else DOWN

    # 1d chart
    fig, ax = plt.subplots(figsize=(16, 7), layout="constrained")
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
            h["date"] = pd.to_datetime(h["date"], errors="coerce")
            h["value"] = pd.to_numeric(h["value"], errors="coerce")
            for days, label in [(7, "7d"), (30, "1m"), (365, "1y")]:
                fig, ax = plt.subplots(figsize=(16, 7), layout="constrained")
                decorate(ax, f"{name} ({label})", "Date", "Index Value")
                hh = h.tail(days)
                if len(hh) >= 2:
                    col = UP if hh["value"].iloc[-1] >= hh["value"].iloc[0] else DOWN
                    ax.plot(hh["date"], hh["value"], lw=2.0, color=col)
                else:
                    ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", alpha=0.5)
                save(fig, os.path.join(OUTDIR, f"{key}_{label}.png"))

    # サイト用の % テキスト
    txt = f"{name} 1d: {(0.0 if delta is None else delta):+0.2f}%"
    with open(os.path.join(OUTDIR, f"{key}_post_intraday.txt"), "w", encoding="utf-8") as f:
        f.write(txt)

if __name__ == "__main__":
    main()
