# -*- coding: utf-8 -*-
"""
R-BANK9 intraday index snapshot
等ウェイト / 前日終値比（%）で1日チャートを描画（黒背景・SNS向け）
最終バー(15:25以降)での異常スパイクを除外。
"""

import os
from typing import List, Dict
from datetime import datetime, timezone, timedelta, time as dtime
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ---------- 設定 ----------
BASE_TZ = timezone(timedelta(hours=9))  # JST
OUT_DIR = "docs/outputs"
TICKER_FILE = "docs/tickers_rbank9.txt"

IMG_PATH = os.path.join(OUT_DIR, "rbank9_intraday.png")
CSV_PATH = os.path.join(OUT_DIR, "rbank9_intraday.csv")
POST_PATH = os.path.join(OUT_DIR, "rbank9_post_intraday.txt")

INTRA_PERIOD = "7d"
INTRA_INTERVAL = "5m"

# 堅牢化パラメータ
COVERAGE_MIN_FRAC = 0.7
MARKET_OPEN_JST = dtime(9, 0)
MARKET_CLOSE_JST = dtime(15, 0)
LAST_BAR_IGNORE_AFTER = dtime(15, 25)  # ← 終盤バー除外の閾値
OUTLIER_ABS_PCT = 10.0                 # ±10%超は外れ値扱い

# ---------- ユーティリティ ----------
def jst_now() -> datetime:
    return datetime.now(BASE_TZ)

def load_tickers(path: str) -> List[str]:
    xs: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            xs.append(s)
    return xs

def ensure_series_1dClose(df: pd.DataFrame) -> pd.Series:
    if "Close" not in df.columns:
        raise ValueError("Close column not found")
    s = pd.to_numeric(df["Close"], errors="coerce").dropna()
    s.index = pd.to_datetime(s.index)
    return s

def _to_jst_indexed(s: pd.Series) -> pd.Series:
    idx = pd.to_datetime(s.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert(BASE_TZ)
    s.index = idx
    return s

def fetch_prev_close(ticker: str) -> float:
    d = yf.download(ticker, period="10d", interval="1d",
                    auto_adjust=False, progress=False)
    if d.empty:
        raise RuntimeError(f"[WARN] prev close empty for {ticker}")
    s = ensure_series_1dClose(d)
    return float(s.iloc[-2]) if len(s) >= 2 else float(s.iloc[-1])

def fetch_intraday_series(ticker: str) -> pd.Series:
    d = yf.download(ticker, period=INTRA_PERIOD, interval=INTRA_INTERVAL,
                    auto_adjust=False, progress=False)
    if d.empty:
        raise RuntimeError(f"[WARN] intraday empty for {ticker}")
    s = ensure_series_1dClose(d)
    s = _to_jst_indexed(s)

    # 当日だけ抽出
    today = s.index[-1].date()
    s = s[s.index.date == today]

    # 取引時間内 (9:00–15:30)
    s = s[(s.index.time >= MARKET_OPEN_JST) & (s.index.time <= MARKET_CLOSE_JST)]

    # 15:25以降のバーを除外
    s = s[s.index.time < LAST_BAR_IGNORE_AFTER]

    if s.empty:
        raise RuntimeError(f"[WARN] intraday filtered empty for {ticker}")
    return s

# ---------- 指数構築 ----------
def build_equal_weight_index(tickers: List[str]) -> pd.DataFrame:
    pct_map: Dict[str, pd.Series] = {}

    for t in tickers:
        try:
            prev = fetch_prev_close(t)
            intraday = fetch_intraday_series(t)
            pct = (intraday / prev - 1.0) * 100.0
            pct = pct.mask(pct.abs() > OUTLIER_ABS_PCT)
            pct_map[t] = pct.rename(t)
        except Exception as e:
            print(f"[WARN] skip {t}  # {e}")

    if not pct_map:
        raise RuntimeError("No intraday data fetched")

    df = pd.concat(pct_map.values(), axis=1).sort_index()
    df = df.ffill()

    # 被覆率フィルタ
    min_cols = max(1, int(len(df.columns) * COVERAGE_MIN_FRAC))
    df = df[df.count(axis=1) >= min_cols]

    # 等ウェイト平均
    df["R_BANK9"] = df.mean(axis=1, skipna=True)
    return df

# ---------- 可視化 ----------
def pick_line_color(series: pd.Series) -> str:
    return "#00e5d7" if len(series) and float(series.iloc[-1]) >= 0 else "#ff4d4d"

def plot_index(df: pd.DataFrame) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    series = df["R_BANK9"]
    c = pick_line_color(series)

    plt.close("all")
    fig = plt.figure(figsize=(16, 9), dpi=160)
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    for sp in ax.spines.values():
        sp.set_color("#444444")
    ax.plot(series.index, series.values, color=c, linewidth=3.0)
    ax.axhline(0, color="#666666", linewidth=1.0)
    ax.tick_params(colors="white")
    ax.set_title(f"R-BANK9 Intraday Snapshot ({jst_now().strftime('%Y/%m/%d %H:%M')})",
                 color="white", fontsize=22, pad=12)
    ax.set_xlabel("Time", color="white")
    ax.set_ylabel("Change vs Prev Close (%)", color="white")
    fig.tight_layout()
    plt.savefig(IMG_PATH, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

def save_csv(df: pd.DataFrame) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(CSV_PATH, encoding="utf-8")

def save_post_text(df: pd.DataFrame, tickers: List[str]) -> None:
    last = float(df["R_BANK9"].iloc[-1])
    sign = "🔺" if last >= 0 else "🔻"
    with open(POST_PATH, "w", encoding="utf-8") as f:
        f.write(
            f"{sign} R-BANK9 日中スナップショット（{jst_now().strftime('%Y/%m/%d %H:%M')}）\n"
            f"{last:+.2f}%（前日終値比）\n"
            f"※ 15:25以降の最終バー除外済 / 構成9銘柄等ウェイト\n"
            f"#地方銀行 #R_BANK9 #日本株\n"
        )

# ---------- メイン ----------
def main():
    tickers = load_tickers(TICKER_FILE)
    print("[INFO] Building R_BANK9 intraday index ...")
    df = build_equal_weight_index(tickers)
    plot_index(df)
    save_csv(df)
    save_post_text(df, tickers)
    print("[INFO] done.")

if __name__ == "__main__":
    main()
