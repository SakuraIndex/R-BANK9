# -*- coding: utf-8 -*-
"""
R-BANK9 intraday index snapshot (robust)

- できれば yfinance の 5分足から等ウェイト指数(前日比%)を作る
- 5分足が取れない/空の時は「デイリーの前日比%」を等ウェイト平均して
  当日のセッション時間(09:00–15:30 JST)にフラットで展開（フェイルセーフ）
- 出力は必ず ts,pct を含む 1 行以上の CSV にする（ヘッダだけを防ぐ）
"""

from __future__ import annotations

import os
from typing import List, Dict, Optional
from datetime import datetime, time, timedelta, timezone

import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# ===== 基本設定 =====
JST = timezone(timedelta(hours=9))
OUT_DIR = "docs/outputs"
TICKER_FILE = "docs/tickers_rbank9.txt"

CSV_PATH = os.path.join(OUT_DIR, "rbank9_intraday.csv")       # ts,pct
PNG_PATH = os.path.join(OUT_DIR, "rbank9_intraday.png")
POST_PATH = os.path.join(OUT_DIR, "rbank9_post_intraday.txt")

SESSION_START = time(9, 0)
SESSION_END   = time(15, 30)
INTRA_INTERVAL = "5min"  # pandas freq 互換（'5m' ではなく '5min'）

# フェイルセーフのためのクリップ（異常値弾き）
PCT_CLIP_LOW  = -20.0
PCT_CLIP_HIGH =  20.0

# ===== ダークテーマ =====
BG = "#0b1220"
FG = "#d1d5db"
GRID = "#334155"


def now_jst() -> datetime:
    return datetime.now(JST)


def read_tickers(path: str) -> List[str]:
    xs: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            xs.append(s)
    return xs


def _ensure_1d_close(df: pd.DataFrame) -> pd.Series:
    if "Close" not in df.columns:
        raise ValueError("Close column not found")
    s = pd.to_numeric(df["Close"], errors="coerce").dropna()
    s.index = pd.to_datetime(df.index)
    return s


def prev_close(ticker: str) -> float:
    d = yf.download(ticker, period="10d", interval="1d", auto_adjust=False, progress=False)
    s = _ensure_1d_close(d)
    if len(s) >= 2:
        return float(s.iloc[-2])
    return float(s.iloc[-1])


def intraday_series(ticker: str) -> pd.Series:
    """
    直近の当日(JST)5分足 Close → 前日比% の Series を返す。
    取れなければ例外。
    """
    d = yf.download(ticker, period="7d", interval="5m", auto_adjust=False, progress=False)
    s = _ensure_1d_close(d)
    if s.empty:
        raise RuntimeError("empty intraday close")

    # index → JST
    idx = pd.to_datetime(s.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert(JST)
    s = pd.Series(s.values, index=idx).sort_index()

    # 当日だけ
    last_day = s.index[-1].date()
    s = s[s.index.date == last_day]
    if s.empty:
        raise RuntimeError("intraday filtered empty")

    pc = prev_close(ticker)
    pct = (s / pc - 1.0) * 100.0
    pct = pct.clip(lower=PCT_CLIP_LOW, upper=PCT_CLIP_HIGH)
    pct.name = ticker
    return pct


def build_equal_weight_intraday(tickers: List[str]) -> pd.Series:
    """可能なら 5分足等ウェイトの前日比%（当日分）"""
    parts: Dict[str, pd.Series] = {}
    ends: List[pd.Timestamp] = []

    for t in tickers:
        try:
            s = intraday_series(t)
            parts[t] = s
            ends.append(s.index.max())
        except Exception as e:
            print(f"[WARN] intraday skip {t}: {e}")

    if not parts:
        return pd.Series(dtype=float)

    # 共通終了時刻までで 5分グリッドに揃える
    common_end = min(ends)
    start = min(s.index.min() for s in parts.values())
    grid = pd.date_range(start=start, end=common_end, freq=INTRA_INTERVAL, tz=JST)

    aligned = []
    for t, s in parts.items():
        aligned.append(s.reindex(grid).ffill())

    df = pd.concat(aligned, axis=1)
    idx_pct = df.mean(axis=1, skipna=True)
    idx_pct.name = "R_BANK9"
    return idx_pct.dropna()


def fallback_flat_from_daily(tickers: List[str]) -> pd.Series:
    """
    5分足が全滅のとき：各銘柄の直近デイリー前日比% を等ウェイト平均し、
    セッション時間のフラット線を返す（必ず少なくとも 1 点以上）。
    """
    vals = []
    for t in tickers:
        try:
            d = yf.download(t, period="10d", interval="1d", auto_adjust=False, progress=False)
            s = _ensure_1d_close(d)
            if len(s) < 2:
                continue
            pct = (s.iloc[-1] / s.iloc[-2] - 1.0) * 100.0
            pct = float(max(min(pct, PCT_CLIP_HIGH), PCT_CLIP_LOW))
            vals.append(pct)
        except Exception as e:
            print(f"[WARN] daily fallback skip {t}: {e}")

    if not vals:
        # 最悪 0.0% を 1 点
        level = 0.0
    else:
        level = float(sum(vals) / len(vals))

    today = now_jst().date()
    start_dt = datetime.combine(today, SESSION_START, tzinfo=JST)
    end_dt   = datetime.combine(today, SESSION_END, tzinfo=JST)
    grid = pd.date_range(start=start_dt, end=end_dt, freq=INTRA_INTERVAL, tz=JST)
    if len(grid) == 0:
        grid = pd.DatetimeIndex([now_jst()])

    s = pd.Series([level] * len(grid), index=grid, name="R_BANK9")
    return s


def to_ts_pct(df_or_s: pd.Series | pd.DataFrame) -> pd.DataFrame:
    if isinstance(df_or_s, pd.Series):
        s = df_or_s
    else:
        # 既に 1 列ならそれを、複数列なら平均
        if df_or_s.shape[1] == 1:
            s = df_or_s.iloc[:, 0]
        else:
            s = df_or_s.mean(axis=1, skipna=True)
    out = pd.DataFrame({"ts": s.index.tz_convert(JST), "pct": s.values})
    return out.dropna().reset_index(drop=True)


def save_csv_ts_pct(d: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # ISO8601(+09:00) で保存
    tmp = d.copy()
    tmp["ts"] = tmp["ts"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    tmp.to_csv(path, index=False, header=["ts", "pct"])


def render_png(d: pd.DataFrame, out: str) -> None:
    plt.rcParams["figure.facecolor"] = BG
    fig, ax = plt.subplots(figsize=(16, 9), dpi=160)
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(True, alpha=0.35, color=GRID, linestyle="-", linewidth=0.7)
    ax.tick_params(colors=FG, labelcolor=FG)
    ax.title.set_color(FG)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)

    title = f"R-BANK9 Intraday Snapshot ({now_jst().strftime('%Y/%m/%d %H:%M JST')})"
    if d.empty:
        ax.set_title(title + " (no data)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Change vs Prev Close (%)")
    else:
        ax.plot(pd.to_datetime(d["ts"]), d["pct"], linewidth=2.0, color="#f87171")
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Change vs Prev Close (%)")

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor=BG, edgecolor=BG)
    plt.close(fig)


def write_post(d: pd.DataFrame, out: str) -> None:
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if d.empty:
        txt = "▲ R-BANK9 日中スナップショット（no data）\n+0.00%（前日終値比）\n#地方銀行  #R_BANK9 #日本株\n"
    else:
        val = float(d["pct"].iloc[-1])
        sign = "+" if val >= 0 else ""
        txt = (
            f"▲ R-BANK9 日中スナップショット（{now_jst().strftime('%Y/%m/%d %H:%M JST')}）\n"
            f"{sign}{val:.2f}%（前日終値比）\n"
            f"※ 構成9銘柄の等ウェイト\n"
            f"#地方銀行  #R_BANK9 #日本株\n"
        )
    with open(out, "w", encoding="utf-8") as f:
        f.write(txt)


def main():
    tickers = read_tickers(TICKER_FILE)

    # 1) まず 5分足ベース
    s = build_equal_weight_intraday(tickers)

    # 2) 空ならデイリーフォールバック
    if s.empty:
        print("[INFO] intraday empty; fallback to daily flat line")
        s = fallback_flat_from_daily(tickers)

    # ts,pct 化（必ず >=1 行）
    d = to_ts_pct(s)
    if d.empty:
        # 念のため最終防御
        d = pd.DataFrame({"ts": [now_jst()], "pct": [0.0]})

    # 出力
    save_csv_ts_pct(d, CSV_PATH)
    render_png(d, PNG_PATH)
    write_post(d, POST_PATH)
    print(f"[INFO] rows={len(d)} written to {CSV_PATH}")


if __name__ == "__main__":
    main()
