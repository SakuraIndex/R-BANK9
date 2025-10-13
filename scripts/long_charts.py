# scripts/long_charts.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re
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

# ---- util ----
def _lower(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def _best_time_col(df: pd.DataFrame) -> str | None:
    """
    全列を走査し「日時として解釈できる率」が最も高い列を time とみなす。
    0.6 未満なら先頭列を time とする。
    """
    if df.empty or df.shape[1] == 0:
        return None
    best, best_rate = None, -1.0
    for c in df.columns:
        s = pd.to_datetime(df[c], errors="coerce", utc=True)
        rate = s.notna().mean()
        if rate > best_rate:
            best, best_rate = c, rate
    if best_rate >= 0.6:
        return best
    # ヘッダが空欄などでも先頭列を time とする
    return df.columns[0]

def read_intraday(path: str) -> pd.DataFrame:
    """
    intraday CSV -> DataFrame(time[JST tz-aware], value[float])
    - 時刻列が無名でもOK
    - '# ...' で始まるコメント列/Unnamed列/空列名を除去
    - 数値列は横持ち（複数ティッカー）想定、平均を value にする
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=["time", "value"])

    # 先頭カンマ（無名 index）に対応：index_col=0 で読む → reset_index で列化
    raw = pd.read_csv(path, dtype=str, index_col=0)
    raw.reset_index(inplace=True)
    raw.rename(columns={"index": "time"}, inplace=True)

    if raw.empty:
        return pd.DataFrame(columns=["time", "value"])

    # 列正規化
    # 1) コメント/Unnamed/空列名の除去（大文字小文字問わず）
    cols = list(raw.columns)
    drop = []
    for c in cols:
        cs = str(c).strip()
        if cs == "" or cs.lower().startswith("unnamed") or cs.startswith("#"):
            drop.append(c)
    if drop:
        raw = raw.drop(columns=drop)

    # 2) 正規化済み小文字名も持つ（後工程の簡便化用）
    df = _lower(raw)

    # 3) 改めてコメント/unnamed/空列名を二重チェックで落とす
    drop2 = [c for c in df.columns if (c == "" or c.startswith("#") or c.startswith("unnamed"))]
    if drop2:
        df = df.drop(columns=drop2)

    if df.empty or df.shape[1] == 0:
        return pd.DataFrame(columns=["time", "value"])

    # 4) time 列の推定
    tcol = _best_time_col(df)
    if tcol is None:
        return pd.DataFrame(columns=["time", "value"])

    # 5) 値カラム推定
    vcol = None
    # 「value / index / score / mean」を優先
    for c in df.columns:
        if c == tcol:
            continue
        if any(k in c for k in ("value", "index", "score", "mean")):
            vcol = c
            break

    # 見つからなければ「数値にできる列」を全回収 → 行平均
    if vcol is None:
        num_cols = []
        for c in df.columns:
            if c == tcol:
                continue
            try:
                pd.to_numeric(df[c])
                num_cols.append(c)
            except Exception:
                pass
        if len(num_cols) >= 1:
            # 1 本でも描く、複数なら平均
            if len(num_cols) == 1:
                vcol = num_cols[0]
            else:
                df["__mean__"] = df[num_cols].apply(
                    lambda row: pd.to_numeric(row, errors="coerce").mean(), axis=1
                )
                vcol = "__mean__"

    if vcol is None:
        # 最低限 2 列以上あれば 2 列目を値にする
        if len(df.columns) >= 2:
            vcol = df.columns[1]
        else:
            return pd.DataFrame(columns=["time", "value"])

    # 6) 時刻（UTC 文字列 or 既に tz 付）→ JST tz-aware へ
    t = pd.to_datetime(df[tcol], errors="coerce", utc=True)
    # すでに tz-aware の場合、utc=True 指定でも UTC 化されるので OK
    out = pd.DataFrame({"time": t.dt.tz_convert(JP)})
    out["value"] = pd.to_numeric(df[vcol], errors="coerce")

    out = out.dropna(subset=["time", "value"]).sort_values("time").reset_index(drop=True)
    return out

def resample(df: pd.DataFrame, rule: str = "1min") -> pd.DataFrame:
    if df.empty:
        return df
    g = df.set_index("time").sort_index()[["value"]].resample(rule).mean()
    g["value"] = g["value"].interpolate(limit_direction="both")
    return g.reset_index()

def pick_window(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """JST/NY/24hの窓取り。空になったら末尾フェイルセーフ。"""
    if df.empty:
        return df
    now_jst = pd.Timestamp.now(tz=JP)
    today = now_jst.normalize()

    w = pd.DataFrame()
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
    else:  # scoin_plus は rolling-24h
        w = df[df["time"] >= (pd.Timestamp.now(tz=JP) - pd.Timedelta(hours=24))]

    if w.empty:
        w = df.tail(600)  # フェイルセーフ
    return w.reset_index(drop=True)

def decide_pct(series_vals) -> float | None:
    """系列の性質から 騰落率(%) を賢く決める（±120% でクリップ）"""
    s = pd.to_numeric(series_vals, errors="coerce").dropna()
    if len(s) < 2:
        return None
    base, last = float(s.iloc[0]), float(s.iloc[-1])
    vmin, vmax = float(s.min()), float(s.max())
    vabs_med = float(s.abs().median())

    CAP = 120.0

    def clip(p):
        if p is None:
            return None
        return max(-CAP, min(CAP, p))

    # 微小レンジ（±0.5付近が多数）なら積み上げ近似（値が収益率っぽいケース）
    if (vmax - vmin) <= 1.0 and vabs_med <= 0.5:
        prod = 1.0
        for v in s.values:
            prod *= (1.0 + float(v))
        return clip((prod - 1.0) * 100.0)

    # 比率が妥当（符号一致 & 基準が十分）なら比
    if abs(base) > 1e-9 and (base * last) > 0:
        return clip(((last / base) - 1.0) * 100.0)

    # それ以外（符号跨ぎ等）は差分（%ポイント近似）
    return clip((last - base) * 100.0)

def decorate(ax, title, xl, yl):
    ax.set_title(title, color=TITLE, fontsize=20, pad=12)
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(True, alpha=GRID_A)
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
    history_csv  = os.path.join(OUTDIR, f"{key}_history.csv")

    try:
        i = read_intraday(intraday_csv)
        i = pick_window(i, key)
        i = resample(i, "1min")
    except Exception as e:
        print("intraday load fail:", e)
        i = pd.DataFrame(columns=["time", "value"])

    delta = decide_pct(i["value"]) if not i.empty else None
    color = UP if (delta is None or delta >= 0) else DOWN

    # --- 1d ---
    fig, ax = plt.subplots(figsize=(16, 7), layout="constrained")
    decorate(ax, f"{name} (1d)", "Time", "Index Value")
    if not i.empty:
        ax.plot(i["time"], i["value"], lw=2.2, color=color)
    else:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", alpha=0.6)
    save(fig, os.path.join(OUTDIR, f"{key}_1d.png"))

    # --- 7d / 1m / 1y（履歴があれば） ---
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
                        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", alpha=0.5)
                    save(fig, os.path.join(OUTDIR, f"{key}_{label}.png"))
        except Exception as e:
            print("history load fail:", e)

    # サイト用の % テキスト
    txt = f"{name} 1d: {(0.0 if delta is None else delta):+0.2f}%"
    with open(os.path.join(OUTDIR, f"{key}_post_intraday.txt"), "w", encoding="utf-8") as f:
        f.write(txt)

if __name__ == "__main__":
    main()
