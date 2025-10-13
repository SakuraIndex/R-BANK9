# scripts/long_charts.py
# -*- coding: utf-8 -*-
"""
Long-term charts generator (1d / 7d / 1m / 1y)

- Robust intraday CSV loader:
  * Accepts both long format:  [time,value(,volume)]
  * And wide format:           [time, 5830.T, 5844.T, ...]  -> uses row-wise mean as value
- Session clamp by index:
  JP (Astra4/R-BANK9) ... 09:00–15:30 JST (fallback: 直近日の同セッション)
  US (AIN-10) ............ 09:30–16:00 NY  (fallback: 直近日の同セッション)
  24h (S-COIN+) .......... 直近24時間
- %変化はデータのスケールに応じた安全な推定 + クリップ
- 出力:
  docs/outputs/<index>_1d.png
  docs/outputs/<index>_7d.png
  docs/outputs/<index>_1m.png
  docs/outputs/<index>_1y.png
  docs/outputs/<index>_post_intraday.txt  （見出し用の±xx.xx%）
"""

from __future__ import annotations
import os, re
from typing import List, Optional

import pandas as pd
import pytz
import matplotlib
import matplotlib.pyplot as plt

# ========= TZ =========
JP = pytz.timezone("Asia/Tokyo")
NY = pytz.timezone("America/New_York")

# ========= Theme =========
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

# %の上限（指数ごと）
CAP_PER_INDEX = {
    "scoin_plus": 35.0,   # 仮想通貨でも極端な誤判定は抑制
    "ain10":      30.0,
    "astra4":     15.0,
    "rbank9":     15.0,
}
DEFAULT_CAP = 60.0  # 予備（他指数が増えた場合）

# ========= util =========
def _lower(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def _find_time_col(cols: List[str]) -> Optional[str]:
    # 代表的な時刻系の名前優先
    for k in ("time", "timestamp", "date", "datetime", "日時", "時刻"):
        if k in cols:
            return k
    # それでも無ければ最初の列を時刻とみなす
    return cols[0] if cols else None

# ========= CSV loader (robust) =========
def read_intraday(path: str) -> pd.DataFrame:
    """
    intraday CSV -> DataFrame[time(JST tz-aware), value(float)]
    - 長形式（time,value）も、銘柄横持ちの広形式（time, 5830.T, 5844.T, ...）も受け付ける
    - 広形式は行方向の平均を代表値(value)として採用
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=["time", "value"])
    raw = pd.read_csv(path, dtype=str)
    if raw.empty:
        return pd.DataFrame(columns=["time", "value"])

    df = _lower(raw)

    # 先頭に # で始まるメタ列が混ざっていたら除去
    drop = [c for c in df.columns if str(c).strip().startswith("#")]
    if drop:
        df = df.drop(columns=drop)

    tcol = _find_time_col(df.columns.tolist())
    if tcol is None:
        return pd.DataFrame(columns=["time", "value"])

    # 値カラムを推定（"value","index","score","mean" を優先）
    vcol = None
    for c in df.columns:
        if c == tcol:
            continue
        if any(k in c for k in ("value", "index", "score", "mean")):
            vcol = c
            break

    # ------ 広形式（銘柄が横並び）のときは行平均を使う ------
    if vcol is None:
        numeric_candidates = []
        for c in df.columns:
            if c == tcol:
                continue
            try:
                pd.to_numeric(df[c])
                numeric_candidates.append(c)
            except Exception:
                pass

        if len(numeric_candidates) >= 2:
            # 行方向に平均を作る（完全NaN行はNaNのまま）
            df["__row_mean__"] = df[numeric_candidates].apply(
                lambda row: pd.to_numeric(row, errors="coerce").mean(), axis=1
            )
            vcol = "__row_mean__"

    # それでも見つからなければ、2列目を値として採用（最後の砦）
    if vcol is None:
        vcol = df.columns[1] if len(df.columns) > 1 else tcol

    # ------ 時刻をtz-aware JSTへ ------
    t = pd.to_datetime(df[tcol], errors="coerce", utc=True)
    if t.dt.tz is None:  # naive → UTCとして解釈
        t = pd.to_datetime(df[tcol], errors="coerce").dt.tz_localize("UTC")
    # もし列名に jst が入っていたら JST 扱いに（既に上でUTC化しているので最終的にJSTへ統一）
    if "jst" in tcol.lower():
        t = t.dt.tz_convert(JP)
    out = pd.DataFrame({"time": t.dt.tz_convert(JP)})

    out["value"] = pd.to_numeric(df[vcol], errors="coerce")
    out = out.dropna(subset=["time", "value"]).sort_values("time").reset_index(drop=True)
    return out

def resample_minutes(df: pd.DataFrame, rule: str = "1min") -> pd.DataFrame:
    if df.empty:
        return df
    tmp = df.set_index("time").sort_index()
    out = tmp[["value"]].resample(rule).mean()
    out["value"] = out["value"].interpolate(limit_direction="both")
    return out.reset_index()

# ========= セッション選択 =========
def pick_session_window(df: pd.DataFrame, key: str) -> pd.DataFrame:
    if df.empty:
        return df
    now_jst = pd.Timestamp.now(tz=JP)
    today = now_jst.normalize()

    def _fallback(prev_days: int, start_hm: str, end_hm: str, tz):
        base = (pd.Timestamp.now(tz=tz).normalize() - pd.Timedelta(days=prev_days))
        s = pd.Timestamp(f"{base.date()} {start_hm}", tz=tz)
        e = pd.Timestamp(f"{base.date()} {end_hm}", tz=tz)
        # df はJSTなので変換して比較
        jj = df["time"].dt.tz_convert(tz)
        m = (jj >= s) & (jj <= e)
        return df.loc[m].reset_index(drop=True)

    if key in ("astra4", "rbank9"):
        # 当日 JST 09:00–15:30
        s = pd.Timestamp(f"{today.date()} 09:00", tz=JP)
        e = pd.Timestamp(f"{today.date()} 15:30", tz=JP)
        m = (df["time"] >= s) & (df["time"] <= e)
        w = df.loc[m]
        if w.empty:
            # 前営業日（厳密な営業日判定はしないが、直近日で再トライ）
            w = _fallback(1, "09:00", "15:30", JP)
        return w.reset_index(drop=True)

    if key == "ain10":
        # 米国 09:30–16:00（NY時間）。データ自体はJSTなのでNYに変換して窓を切る
        jj = df["time"].dt.tz_convert(NY)
        base = pd.Timestamp.now(tz=NY).normalize()
        s = pd.Timestamp(f"{base.date()} 09:30", tz=NY)
        e = pd.Timestamp(f"{base.date()} 16:00", tz=NY)
        m = (jj >= s) & (jj <= e)
        w = df.loc[m]
        if w.empty:
            w = _fallback(1, "09:30", "16:00", NY)
        return w.reset_index(drop=True)

    # scoin_plus（24hローリング）
    from_ = pd.Timestamp.now(tz=JP) - pd.Timedelta(hours=24)
    return df.loc[df["time"] >= from_].reset_index(drop=True)

# ========= %推定 =========
def decide_pct(series_vals: pd.Series, cap: float) -> Optional[float]:
    s = pd.to_numeric(series_vals, errors="coerce").dropna()
    if len(s) < 2:
        return None

    def _clip(p: Optional[float]) -> Optional[float]:
        if p is None:
            return None
        return max(-cap, min(cap, float(p)))

    vmin, vmax = float(s.min()), float(s.max())
    vabs_med = float(s.abs().median())
    base, last = float(s.iloc[0]), float(s.iloc[-1])

    # 小振幅：足ごとの%ポイントとみなし乗算（∏(1+v)-1）
    if (vmax - vmin) <= 1.0 and vabs_med <= 0.5:
        prod = 1.0
        for v in s.values:
            prod *= (1.0 + float(v))
        return _clip((prod - 1.0) * 100.0)

    # ベースが十分離れていて符号一致 → レベル比
    if abs(base) > 1e-9 and (base * last) > 0:
        return _clip(((last / base) - 1.0) * 100.0)

    # それ以外 → 差分を%ポイント換算
    return _clip((last - base) * 100.0)

# ========= plot helper =========
def _decorate(ax, title: str, xl: str, yl: str):
    ax.set_title(title, color=TITLE, fontsize=20, pad=12)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.grid(True, alpha=GRID_A)
    for sp in ax.spines.values():
        sp.set_color(FG)

def _save(fig, path: str):
    fig.savefig(path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)

# ========= main =========
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    key = os.environ.get("INDEX_KEY", "index").strip().lower()
    name = key.upper().replace("_", "")

    intraday_csv = os.path.join(OUTDIR, f"{key}_intraday.csv")
    history_csv  = os.path.join(OUTDIR, f"{key}_history.csv")

    # ----- 1d -----
    try:
        i = read_intraday(intraday_csv)
        i = pick_session_window(i, key)
        i = resample_minutes(i, "1min")
    except Exception as e:
        print(f"[WARN] intraday load failed: {e}")
        i = pd.DataFrame(columns=["time", "value"])

    cap = CAP_PER_INDEX.get(key, DEFAULT_CAP)
    delta = decide_pct(i["value"], cap) if not i.empty else None
    color = UP if (delta is None or delta >= 0) else DOWN

    fig, ax = plt.subplots(figsize=(16, 7), layout="constrained")
    _decorate(ax, f"{name} (1d)", "Time", "Index Value")
    if not i.empty:
        ax.plot(i["time"], i["value"], linewidth=2.4, color=color)
    else:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", alpha=0.6)
    _save(fig, os.path.join(OUTDIR, f"{key}_1d.png"))

    # ----- 7d / 1m / 1y -----
    if os.path.exists(history_csv):
        try:
            h = pd.read_csv(history_csv)
            h = _lower(h)
            if "date" in h.columns and "value" in h.columns:
                h["date"]  = pd.to_datetime(h["date"], errors="coerce")
                h["value"] = pd.to_numeric(h["value"], errors="coerce")
                h = h.dropna(subset=["date", "value"]).sort_values("date")
                for days, label in [(7, "7d"), (30, "1m"), (365, "1y")]:
                    fig, ax = plt.subplots(figsize=(16, 7), layout="constrained")
                    _decorate(ax, f"{name} ({label})", "Date", "Index Value")
                    hh = h.tail(days)
                    if len(hh) >= 2:
                        col = UP if hh["value"].iloc[-1] >= hh["value"].iloc[0] else DOWN
                        ax.plot(hh["date"], hh["value"], linewidth=2.2, color=col)
                    elif len(hh) == 1:
                        ax.plot(hh["date"], hh["value"], marker="o", markersize=6,
                                linewidth=0, color=UP)
                        y = float(hh["value"].iloc[0])
                        ax.set_ylim(y - 0.1, y + 0.1)
                        ax.text(0.5, 0.5, "Only 1 point (need ≥ 2)",
                                transform=ax.transAxes, ha="center", va="center", alpha=0.5)
                    else:
                        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                                ha="center", va="center", alpha=0.5)
                    _save(fig, os.path.join(OUTDIR, f"{key}_{label}.png"))
        except Exception as e:
            print(f"[WARN] history plot failed: {e}")

    # ----- 見出し用 % テキスト -----
    pct_text = "—" if (delta is None) else f"{delta:+0.2f}%"
    with open(os.path.join(OUTDIR, f"{key}_post_intraday.txt"),
              "w", encoding="utf-8") as f:
        f.write(f"{name} 1d: {pct_text}")

    # メモ（任意）
    with open(os.path.join(OUTDIR, "_last_run.txt"), "w", encoding="utf-8") as f:
        f.write(pd.Timestamp.now(tz=JP).isoformat())

if __name__ == "__main__":
    main()
