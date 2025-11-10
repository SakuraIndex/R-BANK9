# -*- coding: utf-8 -*-
"""
R-BANK9 intraday index snapshot
キャッシュ無効化対応版（yfinance の再フェッチを強制）
"""

import os
from typing import List
from datetime import datetime, timezone, timedelta
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

BASE_TZ = timezone(timedelta(hours=9))
OUT_DIR = "docs/outputs"
TICKER_FILE = "docs/tickers_rbank9.txt"

IMG_PATH = os.path.join(OUT_DIR, "rbank9_intraday.png")
CSV_PATH = os.path.join(OUT_DIR, "rbank9_intraday.csv")
POST_PATH = os.path.join(OUT_DIR, "rbank9_post_intraday.txt")

INTRA_PERIOD = "7d"
INTRA_INTERVAL = "5m"

def jst_now():
    return datetime.now(BASE_TZ)

def load_tickers(path):
    with open(path, "r", encoding="utf-8") as f:
        return [s.strip() for s in f if s.strip() and not s.startswith("#")]

def fetch_prev_close(ticker):
    # キャッシュを避けるために unique な引数を渡す
    d = yf.download(
        ticker,
        period="10d",
        interval="1d",
        auto_adjust=True,  # ← TRUE に変更（前日終値の歪みを解消）
        progress=False,
        prepost=False,
        repair=True,  # ← データ欠損補修
        threads=False,
    )
    if d.empty:
        raise RuntimeError(f"[WARN] prev close empty for {ticker}")
    s = pd.to_numeric(d["Close"], errors="coerce").dropna()
    return float(s.iloc[-2]) if len(s) >= 2 else float(s.iloc[-1])

def fetch_intraday_series(ticker):
    # キャッシュ防止のため unique パラメータを付与
    print(f"[INFO] Fetching fresh intraday data for {ticker}")
    d = yf.download(
        ticker,
        period=INTRA_PERIOD,
        interval=INTRA_INTERVAL,
        auto_adjust=True,
        progress=False,
        prepost=False,
        repair=True,
        threads=False,
    )
    if d.empty:
        raise RuntimeError(f"[WARN] intraday empty for {ticker}")

    s = pd.to_numeric(d["Close"], errors="coerce").dropna()
    idx = pd.to_datetime(s.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert(BASE_TZ)
    s.index = idx

    today = idx[-1].date()
    s = s[s.index.date == today]
    s = s[s.index.time < datetime(2025, 11, 10, 15, 25).time()]  # 15:25以降除外

    if s.empty:
        raise RuntimeError(f"[WARN] intraday filtered empty for {ticker}")
    return s

def build_index(tickers):
    parts = []
    for t in tickers:
        try:
            prev = fetch_prev_close(t)
            cur = fetch_intraday_series(t)
            pct = (cur / prev - 1.0) * 100.0
            parts.append(pct.rename(t))
        except Exception as e:
            print(f"[WARN] skip {t} ({e})")

    df = pd.concat(parts, axis=1).sort_index()
    df["R_BANK9"] = df.mean(axis=1, skipna=True)
    return df

def plot(df):
    os.makedirs(OUT_DIR, exist_ok=True)
    s = df["R_BANK9"]
    color = "#00e5d7" if s.iloc[-1] >= 0 else "#ff4d4d"

    plt.figure(figsize=(16, 9), dpi=160)
    plt.style.use("dark_background")
    plt.plot(s.index, s.values, color=color, linewidth=3.0)
    plt.axhline(0, color="#666666", linewidth=1.0)
    plt.title(f"R-BANK9 Intraday Snapshot ({jst_now().strftime('%Y/%m/%d %H:%M')})",
              color="white", fontsize=22, pad=12)
    plt.xlabel("Time (JST)")
    plt.ylabel("Change vs Prev Close (%)")
    plt.tight_layout()
    plt.savefig(IMG_PATH, bbox_inches="tight", facecolor="black")
    plt.close()

def save_post(df):
    s = float(df["R_BANK9"].iloc[-1])
    sign = "🔺" if s >= 0 else "🔻"
    with open(POST_PATH, "w", encoding="utf-8") as f:
        f.write(f"{sign} R-BANK9 日中スナップショット（{jst_now().strftime('%Y/%m/%d %H:%M')}）\n")
        f.write(f"{s:+.2f}%（前日終値比）\n")
        f.write("#地方銀行 #R_BANK9 #日本株\n")

def main():
    tickers = load_tickers(TICKER_FILE)
    df = build_index(tickers)
    df.to_csv(CSV_PATH, encoding="utf-8")
    plot(df)
    save_post(df)
    print("[INFO] done.")

if __name__ == "__main__":
    main()
