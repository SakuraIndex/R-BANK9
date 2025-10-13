# scripts/long_charts.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re, io
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
    "figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": FG,
    "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
    "text.color": FG, "grid.color": FG, "savefig.facecolor": BG,
})

OUTDIR = os.path.join("docs", "outputs")

# 環境で明示された key（なければ repo 名に依存）
def _infer_key() -> str:
    k = os.environ.get("INDEX_KEY", "").strip().lower()
    if k:
        return k
    # repo から推測（Actions で checkout 済み前提）
    try:
        repo = os.environ.get("GITHUB_REPOSITORY", "").split("/")[-1]
        return repo.replace("-", "_").lower() or "index"
    except Exception:
        return "index"

def _lower(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def _find_time_col(cols: pd.Index) -> str | None:
    for c in cols:
        if re.search(r"(?:^|[_\-\s])(time|日時|date|datetime|timestamp|時刻)(?:$|[_\-\s])", str(c), re.I):
            return c
    return cols[0] if len(cols) else None

def _sniff_read(path: str) -> pd.DataFrame:
    """
    区切り文字（カンマ/タブ/セミコロン/スペースなど）を自動判別して読み取る。
    """
    with open(path, "rb") as f:
        head = f.read(4096)
    buf = io.BytesIO(head)
    sample = buf.read().decode("utf-8", errors="ignore")
    # pandas の方に任せて sniff させる
    return pd.read_csv(path, dtype=str, sep=None, engine="python")

def read_intraday(path: str) -> pd.DataFrame:
    """
    intraday CSV -> DataFrame(time[JST tz-aware], value[float])
    値列は次の優先順位で選ぶ：
      1) 列名に index名（key）を含む列（例 R_BANK9 / s-coin+ などの大小無視）
      2) 'value','index','score','mean' 含む列
      3) 数値に変換できる列のうち **末尾列**（多銘柄合成CSVで右端が合成値想定）
      4) だめなら2列目
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=["time", "value"])

    try:
        raw = _sniff_read(path)
    except Exception:
        return pd.DataFrame(columns=["time", "value"])
    if raw.empty:
        return pd.DataFrame(columns=["time", "value"])

    df = _lower(raw)
    tcol = _find_time_col(df.columns)
    if tcol is None:
        return pd.DataFrame(columns=["time", "value"])

    key = _infer_key()
    key_pat = re.sub(r"[^a-z0-9]+", "", key, flags=re.I)

    # 候補列
    cand = [c for c in df.columns if c != tcol]

    # 1) key を含む列（大小/記号無視）
    def _norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", s.lower())

    vcol = None
    for c in cand:
        if key_pat and key_pat in _norm(str(c)):
            vcol = c
            break
    # 2) 意味名
    if vcol is None:
        for c in cand:
            if re.search(r"value|index|score|mean", str(c), re.I):
                vcol = c
                break
    # 3) 数値列の末尾を優先
    if vcol is None:
        numable = []
        for c in cand:
            try:
                pd.to_numeric(df[c])
                numable.append(c)
            except Exception:
                pass
        if numable:
            vcol = numable[-1]
    # 4) どうしても無ければ2列目
    if vcol is None and len(df.columns) >= 2:
        vcol = df.columns[1]

    # ---- 時刻 tz-aware(JST) へ ----
    t = pd.to_datetime(df[tcol], errors="coerce", utc=True)
    # tz 無しなら UTC とみなす
    if t.dt.tz is None:
        t = pd.to_datetime(df[tcol], errors="coerce").dt.tz_localize("UTC")
    out = pd.DataFrame({"time": t.dt.tz_convert(JP)})

    # 値
    out["value"] = pd.to_numeric(df[vcol], errors="coerce")
    out = out.dropna(subset=["time", "value"]).sort_values("time").reset_index(drop=True)
    print(f"[read_intraday] path={path} rows={len(out)} vcol={vcol}")
    return out

def resample(df: pd.DataFrame, rule="1min") -> pd.DataFrame:
    if df.empty:
        return df
    g = df.set_index("time").sort_index()[["value"]].resample(rule).mean()
    g["value"] = g["value"].interpolate(limit_direction="both")
    return g.reset_index()

def pick_window(df: pd.DataFrame, key: str, kind: str) -> pd.DataFrame:
    """
    kind: 'session' (JST 9:00-15:30), 'sessionUS' (NY 9:30-16:00), 'rolling24'
    フォールバック: 直前営業日窓 or tail(600)
    """
    if df.empty:
        return df

    now_jst = pd.Timestamp.now(tz=JP)
    if kind == "rolling24":
        from_ts = now_jst - pd.Timedelta(hours=24)
        win = df[(df["time"] >= from_ts) & (df["time"] <= now_jst)]
        if len(win) < 10:
            win = df.tail(600)
        print(f"[window] {key} rolling24 win_len={len(win)}")
        return win.reset_index(drop=True)

    # セッション（JST）
    if kind in ("session", "sessionUS"):
        if kind == "session":
            day = now_jst.normalize()
            s = pd.Timestamp(f"{day.date()} 09:00", tz=JP)
            e = pd.Timestamp(f"{day.date()} 15:30", tz=JP)
            win = df[(df["time"] >= s) & (df["time"] <= e)]
            if len(win) == 0:
                y = (day - pd.Timedelta(days=1)).date()
                s2 = pd.Timestamp(f"{y} 09:00", tz=JP)
                e2 = pd.Timestamp(f"{y} 15:30", tz=JP)
                win = df[(df["time"] >= s2) & (df["time"] <= e2)]
        else:
            # US セッション（NY）
            tny = df["time"].dt.tz_convert(NY)
            day = pd.Timestamp.now(tz=NY).normalize()
            s = pd.Timestamp(f"{day.date()} 09:30", tz=NY)
            e = pd.Timestamp(f"{day.date()} 16:00", tz=NY)
            win = df[(tny >= s) & (tny <= e)]
            if len(win) == 0:
                y = (day - pd.Timedelta(days=1)).date()
                s2 = pd.Timestamp(f"{y} 09:30", tz=NY)
                e2 = pd.Timestamp(f"{y} 16:00", tz=NY)
                win = df[(tny >= s2) & (tny <= e2)]

        if len(win) == 0:
            win = df.tail(600)
        print(f"[window] {key} {kind} win_len={len(win)}")
        return win.reset_index(drop=True)

    # デフォルト
    win = df.tail(600)
    print(f"[window] {key} default win_len={len(win)}")
    return win.reset_index(drop=True)

def decide_pct(series_vals: pd.Series) -> float | None:
    """
    系列が「リターンの積み上げ（小さい値が並ぶ）」か
    「指数レベル（絶対値が0.5以上が普通）」かをだいたい判別して騰落率(%)。
    クリップは緩め（±200%）で S-COIN+ の -30% 固定問題を排除。
    """
    s = pd.to_numeric(series_vals, errors="coerce").dropna()
    if len(s) < 2:
        return None

    CAP = 200.0
    vmin, vmax = float(s.min()), float(s.max())
    vabs_med = float(s.abs().median())
    first, last = float(s.iloc[0]), float(s.iloc[-1])

    # 小さい値中心 → リターン系列（∏(1+v) - 1）
    if (vmax - vmin) <= 1.0 and vabs_med <= 0.5:
        prod = 1.0
        for v in s.values:
            prod *= (1.0 + float(v))
        pct = (prod - 1.0) * 100.0
    else:
        # 指数レベル → 前日比近似（最後/最初 - 1）
        if abs(first) < 1e-9:
            return None
        pct = ((last / first) - 1.0) * 100.0

    if pct > CAP: pct = CAP
    if pct < -CAP: pct = -CAP
    return pct

def decorate(ax, title, xl, yl):
    ax.set_title(title, color=TITLE, fontsize=20, pad=12)
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(True, alpha=GRID_A)
    for s in ax.spines.values(): s.set_color(FG)

def save(fig, path):
    fig.savefig(path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    key = _infer_key()          # e.g. 'scoin_plus', 'rbank9'
    name = key.upper().replace("_", "")

    intraday_csv = os.path.join(OUTDIR, f"{key}_intraday.csv")
    history_csv  = os.path.join(OUTDIR, f"{key}_history.csv")

    # どの窓で描くかをキーに応じて決める
    kind = "session"   # JPN デフォルト
    if key == "scoin_plus":
        kind = "rolling24"
    elif key == "ain10":
        kind = "sessionUS"

    try:
        i = read_intraday(intraday_csv)
        i = resample(i, "1min")
        i = pick_window(i, key, kind)
    except Exception as e:
        print("intraday load fail:", e)
        i = pd.DataFrame(columns=["time", "value"])

    delta = decide_pct(i["value"]) if not i.empty else None
    color = UP if (delta is None or delta >= 0) else DOWN

    # --- 1d PNG ---
    fig, ax = plt.subplots(figsize=(16,7), layout="constrained")
    decorate(ax, f"{name} (1d)", "Time", "Index Value")
    if not i.empty:
        ax.plot(i["time"], i["value"], lw=2.2, color=color)
    else:
        ax.text(0.5,0.5,"No data", transform=ax.transAxes, ha="center", va="center", alpha=0.6)
    save(fig, os.path.join(OUTDIR, f"{key}_1d.png"))

    # --- 7d / 1m / 1y PNG ---
    if os.path.exists(history_csv):
        try:
            h = pd.read_csv(history_csv)
            if {"date","value"}.issubset(set(map(str.lower, h.columns))):
                # 念のため列名を合わせる
                cols = {c.lower(): c for c in h.columns}
                h["date"]  = pd.to_datetime(h[cols["date"]], errors="coerce")
                h["value"] = pd.to_numeric(h[cols["value"]], errors="coerce")
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
            print("history plot fail:", e)

    # --- サイト表示用の % テキスト ---
    txt = f"{name} 1d: {(0.0 if delta is None else delta):+0.2f}%"
    with open(os.path.join(OUTDIR, f"{key}_post_intraday.txt"), "w", encoding="utf-8") as f:
        f.write(txt)

if __name__ == "__main__":
    main()
