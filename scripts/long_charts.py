# scripts/long_charts.py  — R-BANK9
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re, io
import pandas as pd
import pytz
import matplotlib
import matplotlib.pyplot as plt

# ========= TZ =========
JP = pytz.timezone("Asia/Tokyo")
NY = pytz.timezone("America/New_York")

# ========= Theme =========
BG, FG, TITLE = "#0E1117", "#E6E6E6", "#f2b6c6"
UP, DOWN, GRID_A = "#3bd6c6", "#ff6b6b", 0.25  # up=青緑, down=赤

matplotlib.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": FG,
    "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
    "text.color": FG, "grid.color": FG, "savefig.facecolor": BG,
})

OUTDIR = os.path.join("docs", "outputs")
os.makedirs(OUTDIR, exist_ok=True)

# ========== small utils ==========
def _lower(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def _find_time_col(cols) -> str | None:
    cols = list(cols)
    if len(cols) == 0:
        return None
    for c in cols:
        if re.search(r"(time|日時|date|datetime|timestamp|時刻)", str(c), re.I):
            return c
    # “time” が無いなら 先頭列を time にみなす（インデックス風CSV対策）
    return cols[0]

def _to_datetime_jst(ser: pd.Series) -> pd.Series:
    # 文字列 → tz-aware（UTC/JST入り混じりを吸収）
    t = pd.to_datetime(ser, errors="coerce", utc=True)
    if t.dt.tz is None:
        # pandasの挙動でここは基本来ないが保険
        t = pd.to_datetime(ser, errors="coerce").dt.tz_localize("UTC")
    return t.dt.tz_convert(JP)

def _rowwise_mean_numeric(df: pd.DataFrame) -> pd.Series:
    # 複数銘柄列を等加重平均（文字列を数値化し NaN 無視して平均）
    num = df.apply(pd.to_numeric, errors="coerce")
    return num.mean(axis=1, skipna=True)

# ========== IO ==========
def read_intraday(path: str) -> pd.DataFrame:
    """
    intraday のCSVを robust に単一系列へ正規化:
    - ヘッダー行欠落や “最初の行が列名化” 問題を自動補正
    - time列をJST tz-awareへ
    - 値列が複数ある場合は等加重平均
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=["time", "value"])

    # 1) 通常読み取り（header=0）
    try:
        raw = pd.read_csv(path, dtype=str)
    except Exception:
        raw = pd.DataFrame()

    # “最初の行が列名になっている” パターンを判定
    header_broken = False
    if not raw.empty:
        # 先頭行が “YYYY-..” 始まり かつ time 系列が見つからないなら壊れヘッダーとみなす
        first_row_join = ",".join(map(str, list(raw.iloc[0].values))) if len(raw) > 0 else ""
        has_time_col = any(re.search(r"(time|日時|date|datetime|timestamp|時刻)", str(c), re.I) for c in raw.columns)
        if (not has_time_col) and re.match(r"\d{4}-\d{2}-\d{2}", first_row_join):
            header_broken = True

    # 2) 壊れヘッダーなら header=None で再読込し、先頭列を time に、残りを値列に
    if raw.empty or header_broken:
        raw = pd.read_csv(path, dtype=str, header=None)
        if raw.empty:
            return pd.DataFrame(columns=["time", "value"])
        # 先頭列を time に、他は値
        raw.columns = ["time"] + [f"v{i}" for i in range(1, raw.shape[1])]
        tcol = "time"
        vcols = [c for c in raw.columns if c != "time"]
        out = pd.DataFrame({
            "time": _to_datetime_jst(raw[tcol]),
            "value": _rowwise_mean_numeric(raw[vcols])
        })
        out = out.dropna(subset=["time", "value"]).sort_values("time").reset_index(drop=True)
        return out

    # 3) 正常ヘッダーケース
    df = _lower(raw)
    tcol = _find_time_col(df.columns)
    if tcol is None:
        return pd.DataFrame(columns=["time", "value"])

    # 値列推定：time以外の列を候補に
    vcols = [c for c in df.columns if c != tcol]
    if len(vcols) == 0:
        # 列が1つしかない（インデックスだけ）場合は空
        return pd.DataFrame(columns=["time", "value"])

    # 等加重平均（行方向）
    value = _rowwise_mean_numeric(df[vcols])
    out = pd.DataFrame({
        "time": _to_datetime_jst(df[tcol]),
        "value": value
    })
    out = out.dropna(subset=["time", "value"]).sort_values("time").reset_index(drop=True)
    return out

def read_history(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["date", "value"])
    h = pd.read_csv(path, dtype=str)
    if h.empty:
        return pd.DataFrame(columns=["date", "value"])
    h = _lower(h)
    # date / value のよくある名寄せ
    dcol = None
    for c in h.columns:
        if re.search(r"(date|day|日時|time)", c, re.I):
            dcol = c; break
    if dcol is None:
        dcol = h.columns[0]
    vcol = None
    for c in h.columns:
        if c == dcol: continue
        if re.search(r"(value|index|mean|score)", c, re.I):
            vcol = c; break
    if vcol is None and len(h.columns) > 1:
        vcol = h.columns[1]
    elif vcol is None:
        return pd.DataFrame(columns=["date", "value"])

    out = pd.DataFrame({
        "date": pd.to_datetime(h[dcol], errors="coerce"),
        "value": pd.to_numeric(h[vcol], errors="coerce")
    }).dropna().sort_values("date").reset_index(drop=True)
    return out

# ========== calc / window ==========
def pick_session_window(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """JSTの 09:00-15:30 を今日→無ければ昨日で抽出。"""
    if df.empty:
        return df
    now_jst = pd.Timestamp.now(tz=JP)
    base = now_jst.normalize()
    s1 = pd.Timestamp(f"{base.date()} 09:00", tz=JP)
    e1 = pd.Timestamp(f"{base.date()} 15:30", tz=JP)
    w = df[(df["time"] >= s1) & (df["time"] <= e1)]
    if w.empty:
        y = base - pd.Timedelta(days=1)
        s2 = pd.Timestamp(f"{y.date()} 09:00", tz=JP)
        e2 = pd.Timestamp(f"{y.date()} 15:30", tz=JP)
        w = df[(df["time"] >= s2) & (df["time"] <= e2)]
    return w.reset_index(drop=True)

def resample_1min(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    g = df.set_index("time").sort_index()[["value"]].resample("1min").mean()
    g["value"] = g["value"].interpolate(limit_direction="both")
    return g.reset_index()

CAPS = {"astra4": 30.0, "rbank9": 30.0, "ain10": 30.0, "scoin_plus": 30.0}

def _clip(p: float, cap: float) -> float:
    if p is None or not pd.notna(p):
        return None
    return max(-cap, min(cap, float(p)))

def calc_percent(series: pd.Series, key: str) -> float | None:
    """値が「レベル or %ポイント」の混在に耐性を持たせて騰落率に落とす。"""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 2:
        return None
    first, last = float(s.iloc[0]), float(s.iloc[-1])
    # 1) level想定: 比率
    pct_ratio = ((last - first) / (abs(first) if abs(first) > 1e-9 else 1.0)) * 100.0
    # 2) %ポイント想定: 単純差
    pct_points = (last - first) * 100.0
    # “より小さい方” を採用（暴れにくい）
    use = pct_points if abs(pct_points) < abs(pct_ratio) else pct_ratio
    return _clip(use, CAPS.get(key, 30.0))

# ========== plot ==========
def decorate(ax, title: str, xl: str, yl: str):
    ax.set_title(title, color=TITLE, fontsize=20, pad=12)
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(True, alpha=GRID_A)
    for s in ax.spines.values(): s.set_color(FG)

def save(fig, path: str):
    fig.savefig(path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)

# ========== main ==========
def main():
    key = os.environ.get("INDEX_KEY", "rbank9").strip().lower()
    name = key.upper()
    intraday_csv = os.path.join(OUTDIR, f"{key}_intraday.csv")
    history_csv  = os.path.join(OUTDIR, f"{key}_history.csv")

    # intraday 読み込み→セッション抽出→1分化
    try:
        i = read_intraday(intraday_csv)
        i = pick_session_window(i, key)
        i = resample_1min(i)
    except Exception as e:
        print("intraday load fail:", e)
        i = pd.DataFrame(columns=["time", "value"])

    # 騰落率
    delta = calc_percent(i["value"], key) if not i.empty else None
    color = UP if (delta is not None and delta >= 0) else DOWN

    # 1d PNG
    fig, ax = plt.subplots(figsize=(16,7), layout="constrained")
    decorate(ax, f"{name} (1d)", "Time", "Index Value")
    if not i.empty:
        ax.plot(i["time"], i["value"], lw=2.2, color=color)
    else:
        ax.text(0.5,0.5,"No data", transform=ax.transAxes, ha="center", va="center", alpha=0.6)
    save(fig, os.path.join(OUTDIR, f"{key}_1d.png"))

    # 7d/1m/1y（ヒストリがあれば）
    h = read_history(history_csv)
    for days, label in [(7,"7d"),(30,"1m"),(365,"1y")]:
        fig, ax = plt.subplots(figsize=(16,7), layout="constrained")
        decorate(ax, f"{name} ({label})", "Date", "Index Value")
        if not h.empty:
            hh = h.tail(days)
            if len(hh) >= 2:
                col = UP if hh["value"].iloc[-1] >= hh["value"].iloc[0] else DOWN
                ax.plot(hh["date"], hh["value"], lw=2.0, color=col)
            else:
                ax.text(0.5,0.5,"No data", transform=ax.transAxes, ha="center", va="center", alpha=0.5)
        else:
            ax.text(0.5,0.5,"No data", transform=ax.transAxes, ha="center", va="center", alpha=0.5)
        save(fig, os.path.join(OUTDIR, f"{key}_{label}.png"))

    # %テキスト（サイトで使用）
    txt = f"{name} 1d: {(0.0 if delta is None else delta):+0.2f}%"
    with open(os.path.join(OUTDIR, f"{key}_post_intraday.txt"), "w", encoding="utf-8") as f:
        f.write(txt)

if __name__ == "__main__":
    main()
