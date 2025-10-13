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

# 騰落率の異常値クリップ（指数ごとに少しだけ幅を変える）
PCT_CLIP = {
    "scoin_plus": 35.0,
    "ain10": 25.0,
    "astra4": 20.0,
    "rbank9": 20.0,
}

# ---------- util ----------
def _lower(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def _find_time_col(cols: list[str]) -> str | None:
    for c in cols:
        if re.search(r"time|日時|date|datetime|timestamp|時刻", c):
            return c
    return cols[0] if cols else None

def _to_datetime_jst(series: pd.Series, colname: str) -> pd.Series:
    # 文字列→時刻（tz-aware JST）
    t = pd.to_datetime(series, errors="coerce", utc=True)
    # もし tz が付いてなければ UTC とみなす
    if getattr(t.dt, "tz", None) is None:
        t = pd.to_datetime(series, errors="coerce").dt.tz_localize("UTC")
    # 列名に jst が含まれていれば JST 認識として扱う（最終的にJSTに統一）
    return t.dt.tz_convert(JP)

def read_intraday(path: str) -> pd.DataFrame:
    """
    intraday CSV を読み、汎用フォーマット DataFrame(time[JST tz-aware], value[float]) を返す。
    値カラムが複数（例：R-BANK9 の銘柄別列）の場合は、**行方向平均**で単一系列にする。
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=["time", "value"])

    raw = pd.read_csv(path, dtype=str)
    if raw.empty:
        return pd.DataFrame(columns=["time", "value"])

    df = _lower(raw)
    tcol = _find_time_col(list(df.columns))
    if tcol is None:
        return pd.DataFrame(columns=["time", "value"])

    # 時刻
    t_jst = _to_datetime_jst(df[tcol], tcol)

    # 値候補列（time 以外）
    cand_cols = [c for c in df.columns if c != tcol]
    if not cand_cols:
        return pd.DataFrame(columns=["time", "value"])

    # 数値化できる列だけに絞る
    num_df_list = []
    for c in cand_cols:
        num = pd.to_numeric(df[c], errors="coerce")
        # 数がほぼ全部NaNなら捨てる
        if num.notna().sum() >= max(3, int(0.1 * len(num))):  # 1割以上が数字なら残す
            num_df_list.append(num)

    if not num_df_list:
        # どうしても見つからない場合は最初の列を数値化して使う
        num_df_list = [pd.to_numeric(df[cand_cols[0]], errors="coerce")]

    # 列が複数ある場合は**平均**（指数の等加重と整合）
    if len(num_df_list) == 1:
        value = num_df_list[0]
    else:
        value = pd.concat(num_df_list, axis=1).mean(axis=1)

    out = pd.DataFrame({"time": t_jst, "value": value})
    out = out.dropna(subset=["time", "value"]).sort_values("time").reset_index(drop=True)
    return out

def resample(df: pd.DataFrame, rule: str = "1min") -> pd.DataFrame:
    if df.empty:
        return df
    g = df.set_index("time").sort_index()[["value"]].resample(rule).mean()
    g["value"] = g["value"].interpolate(limit_direction="both")
    return g.reset_index()

def pick_window(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """指数ごとの“当日ウィンドウ”選択。空なら最後の数百点を返す。"""
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
    else:  # scoin_plus（24h ローリング）
        w = df[df["time"] >= (pd.Timestamp.now(tz=JP) - pd.Timedelta(hours=24))]

    if w.empty:
        # 直近日のセッションにデータなし → 末尾から数百点で描画（予防）
        return df.tail(600).reset_index(drop=True)
    return w.reset_index(drop=True)

def decide_pct(series_vals: pd.Series, key: str) -> float | None:
    """
    系列値から騰落率(%)を安定推定。
    値が ±10 以内なら「%ポイント系」とみなし (last - base) * 100。
    それ以上に広いなら倍率系 ((last/base - 1) * 100) を採用。
    """
    s = pd.to_numeric(series_vals, errors="coerce").dropna()
    if len(s) < 2:
        return None
    base, last = float(s.iloc[0]), float(s.iloc[-1])
    vmin, vmax = float(s.min()), float(s.max())

    cap = PCT_CLIP.get(key, 25.0)

    def clip(p: float) -> float:
        return max(-cap, min(cap, p))

    width = abs(vmax - vmin)
    if width <= 10.0:
        pct = (last - base) * 100.0
    elif abs(base) > 1e-9 and (base * last) > 0:
        pct = ((last / base) - 1.0) * 100.0
    else:
        pct = (last - base) * 100.0

    return clip(pct)

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

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    key = os.environ.get("INDEX_KEY", "index").strip().lower()
    name = key.upper().replace("_", "")

    intraday_csv = os.path.join(OUTDIR, f"{key}_intraday.csv")
    history_csv = os.path.join(OUTDIR, f"{key}_history.csv")

    # intraday を強靭に読み込む
    try:
        i = read_intraday(intraday_csv)
        i = pick_window(i, key)
        i = resample(i, "1min")
    except Exception as e:
        print("intraday load fail:", e)
        i = pd.DataFrame(columns=["time", "value"])

    # 騰落率と色（実値の変化も併用して決定）
    delta = decide_pct(i["value"], key) if not i.empty else None
    if not i.empty:
        real_change = float(i["value"].iloc[-1] - i["value"].iloc[0])
    else:
        real_change = 0.0
    color = UP if (delta is None or (delta >= 0 and real_change >= 0)) else DOWN

    # ---- 1d ----
    fig, ax = plt.subplots(figsize=(16, 7), layout="constrained")
    decorate(ax, f"{name} (1d)", "Time", "Index Value")
    if not i.empty:
        ax.plot(i["time"], i["value"], lw=2.2, color=color)
    else:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", alpha=0.6)
    save(fig, os.path.join(OUTDIR, f"{key}_1d.png"))

    # ---- 7d / 1m / 1y ----
    if os.path.exists(history_csv):
        try:
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
                        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                                ha="center", va="center", alpha=0.5)
                    save(fig, os.path.join(OUTDIR, f"{key}_{label}.png"))
        except Exception as e:
            print("history load/plot fail:", e)

    # ---- サイト用の % テキスト（intraday優先）----
    txt_value = 0.0 if delta is None else float(delta)
    with open(os.path.join(OUTDIR, f"{key}_post_intraday.txt"),
              "w", encoding="utf-8") as f:
        f.write(f"{name} 1d: {txt_value:+0.2f}%")

if __name__ == "__main__":
    main()
