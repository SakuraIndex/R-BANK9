# -*- coding: utf-8 -*-
"""
R-BANK9 intraday index snapshot (equal-weight, vs prev close, percent)

- 9 銘柄を等ウェイトで合成
- 前日終値比（%）を 5 分足で算出
- 直近の取引日（JST）だけを抽出
- 共通 5 分グリッドに reindex + ffill で整列
- クリップで異常値を抑制
- 出力: docs/outputs/rbank9_intraday.csv (ts,pct)
        docs/outputs/rbank9_intraday.png（デバッグ用）
        docs/outputs/rbank9_post_intraday.txt（簡易ポスト文）
"""

from __future__ import annotations

import os
from typing import List, Dict, Tuple
from datetime import datetime, timezone, timedelta, time

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ---------- 設定 ----------
JST = timezone(timedelta(hours=9))
OUT_DIR = "docs/outputs"
TICKER_FILE = "docs/tickers_rbank9.txt"   # 5830.T などを 1 行 1 ティッカー

CSV_PATH  = os.path.join(OUT_DIR, "rbank9_intraday.csv")     # ts,pct
IMG_PATH  = os.path.join(OUT_DIR, "rbank9_intraday.png")     # デバッグ用
POST_PATH = os.path.join(OUT_DIR, "rbank9_post_intraday.txt")

# JP は 1m が不安定なことがあるので 5m
INTRA_INTERVAL = "5m"
INTRA_PERIOD   = "3d"   # 直近 3 営業日

# 安全弁（%）
PCT_CLIP_LOW  = -20.0
PCT_CLIP_HIGH =  20.0

# 市場の見た目の時間（JST）
SESSION_START = time(9, 0)   # 09:00
SESSION_END   = time(15, 30) # 15:30


# ---------- ユーティリティ ----------
def jst_now() -> datetime:
    return datetime.now(JST)


def load_tickers(path: str) -> List[str]:
    """
    行内コメント「# ...」や余分な空白を除去して先頭トークンだけをティッカーとして採用。
    例) "5830.T  # いよぎん" → "5830.T"
    """
    xs: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            s = s.split("#", 1)[0].strip()
            if not s:
                continue
            s = s.split()[0].strip().strip(",;")
            if s:
                xs.append(s)
    if not xs:
        raise RuntimeError("No tickers in docs/tickers_rbank9.txt")
    return xs


def _to_series_1d_close(df: pd.DataFrame) -> pd.Series:
    if "Close" not in df.columns:
        raise ValueError("Close column not found")
    close = df["Close"]
    if isinstance(close, pd.Series):
        return pd.to_numeric(close, errors="coerce").dropna()
    d = close.apply(pd.to_numeric, errors="coerce")
    mask = d.notna().any(axis=0)
    d = d.loc[:, mask]
    if d.shape[1] == 0:
        raise ValueError("no numeric close column")
    s = d.iloc[:, 0] if d.shape[1] == 1 else d[d.count(axis=0).idxmax()]
    return s.dropna().astype(float)


def last_trading_day(ts_index: pd.DatetimeIndex) -> datetime.date:
    idx = pd.to_datetime(ts_index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert(JST)
    return idx[-1].date()


def fetch_prev_close(ticker: str, day: datetime.date) -> float:
    d = yf.download(ticker, period="10d", interval="1d", auto_adjust=False, progress=False)
    if d.empty:
        raise RuntimeError(f"prev close empty for {ticker}")
    s = _to_series_1d_close(d)
    s.index = pd.to_datetime(s.index)
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    s = s.tz_convert(JST)
    s_before = s[s.index.date < day]
    return float((s_before.iloc[-1] if not s_before.empty else s.iloc[-1]))


def fetch_intraday_series(ticker: str) -> pd.Series:
    d = yf.download(ticker, period=INTRA_PERIOD, interval=INTRA_INTERVAL,
                    auto_adjust=False, progress=False)
    if d.empty:
        raise RuntimeError(f"intraday empty for {ticker}")
    s = _to_series_1d_close(d)
    idx = pd.to_datetime(s.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert(JST)
    return pd.Series(s.values, index=idx)


def day_grid(day: datetime.date) -> pd.DatetimeIndex:
    start = pd.Timestamp.combine(day, SESSION_START).tz_localize(JST)
    end   = pd.Timestamp.combine(day, SESSION_END).tz_localize(JST)
    return pd.date_range(start=start, end=end, freq=INTRA_INTERVAL, tz=JST)


def build_equal_weight_pct(tickers: List[str]) -> Tuple[pd.Series, datetime.date]:
    probe = fetch_intraday_series(tickers[0])
    day = last_trading_day(probe.index)
    grid = day_grid(day)
    indiv_pct: Dict[str, pd.Series] = {}

    def _slice_day(s: pd.Series) -> pd.Series:
        x = s[(s.index.date == day)]
        if x.empty:
            d2 = last_trading_day(s.index)
            x = s[(s.index.date == d2)]
        return x

    for t in tickers:
        try:
            intraday = _slice_day(fetch_intraday_series(t))
            if intraday.empty:
                print(f"[WARN] {t}: no intraday for target day, skip")
                continue
            prev = fetch_prev_close(t, day)
            pct = (intraday / prev - 1.0) * 100.0
            pct = pct.clip(lower=PCT_CLIP_LOW, upper=PCT_CLIP_HIGH)
            pct = pct.reindex(grid).ffill()
            indiv_pct[t] = pct.rename(t)
        except Exception as e:
            print(f"[WARN] skip {t}  # {e}")

    if not indiv_pct:
        return pd.Series(0.0, index=grid, name="R_BANK9"), day

    df = pd.concat(indiv_pct.values(), axis=1)
    series = df.mean(axis=1, skipna=True).astype(float)
    series = series.ffill().fillna(0.0)
    series.name = "R_BANK9"
    return series, day


def save_ts_pct_csv(series: pd.Series, day: datetime.date, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if s.empty:
        s = pd.Series(0.0, index=day_grid(day))
    idx = pd.DatetimeIndex(s.index).tz_convert(JST)
    ts_list = [ts.isoformat() for ts in idx.to_pydatetime()]
    df = pd.DataFrame({"ts": ts_list, "pct": s.round(4)})

    # --- 強制更新: JST現時刻行を追加 ---
    now_ts = jst_now().isoformat()
    df.loc[len(df)] = [now_ts, 0.0]

    df.to_csv(path, index=False)
    print(f"[INFO] CSV saved ({len(df)} rows) → {path}")


def plot_debug(series: pd.Series, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.close("all")
    fig, ax = plt.subplots(figsize=(14, 6), dpi=140)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    for sp in ax.spines.values():
        sp.set_color("#333333")
    ax.grid(True, color="#2a2a2a", alpha=0.5, linestyle="--", linewidth=0.7)

    ax.plot(series.index, series.values, color="#f87171", linewidth=2.0)
    ax.axhline(0, color="#666666", linewidth=1.0)
    ax.tick_params(colors="white")
    ax.set_title(f"R-BANK9 Intraday Snapshot ({jst_now().strftime('%Y/%m/%d %H:%M JST')})",
                 color="white", fontsize=16, pad=10)
    ax.set_xlabel("Time", color="white")
    ax.set_ylabel("Change vs Prev Close (%)", color="white")

    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_post(series: pd.Series, path: str) -> None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    last = float(s.iloc[-1]) if not s.empty else 0.0
    sign = "+" if last >= 0 else ""
    text = (
        f"▲ R-BANK9 日中スナップショット（{jst_now().strftime('%Y/%m/%d %H:%M JST')}）\n"
        f"{sign}{last:.2f}%（前日終値比）\n"
        f"※ 構成9銘柄の等ウェイト\n"
        f"#地方銀行  #R_BANK9 #日本株\n"
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ---------- メイン ----------
def main():
    tickers = load_tickers(TICKER_FILE)
    print(f"[INFO] tickers: {', '.join(tickers)}")
    print("[INFO] building equal-weight percent series...")
    series, day = build_equal_weight_pct(tickers)

    # 出力
    save_ts_pct_csv(series, day, CSV_PATH)
    plot_debug(series, IMG_PATH)
    save_post(series, POST_PATH)

    print("[INFO] done.")
    print(f"[INFO] wrote: {CSV_PATH}, {IMG_PATH}, {POST_PATH}")
    tail_n = min(5, len(series))
    print("[INFO] tail:")
    print(pd.DataFrame({"ts": pd.DatetimeIndex(series.index)[-tail_n:], "pct": series[-tail_n:]}))


if __name__ == "__main__":
    main()
