# -*- coding: utf-8 -*-
"""
R-BANK9 intraday index snapshot
等ウェイト / 前日終値比（%）で1日チャートを描画（黒背景・SNS向け）
"""

import os
from typing import List
from datetime import datetime, timezone, timedelta

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ---------- 設定 ----------
BASE_TZ = timezone(timedelta(hours=9))  # JST
OUT_DIR = "docs/outputs"
TICKER_FILE = "docs/tickers_rbank9.txt"   # 1行1ティッカー（例: 5830.T）

IMG_PATH = os.path.join(OUT_DIR, "rbank9_intraday.png")
CSV_PATH = os.path.join(OUT_DIR, "rbank9_intraday.csv")
POST_PATH = os.path.join(OUT_DIR, "rbank9_post_intraday.txt")

# yfinance の JP 現物は 1m が不安定なことがあるので 5m を既定に
INTRA_PERIOD = "7d"
INTRA_INTERVAL = "5m"   # 1mで動くなら "1d"+"1m" でもOK

# ---------- ユーティリティ ----------
def jst_now() -> datetime:
    return datetime.now(BASE_TZ)

def load_tickers(path: str) -> List[str]:
    tickers: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            tickers.append(line)
    return tickers

def ensure_series_1dClose(df: pd.DataFrame) -> pd.Series:
    """
    df['Close'] を「1次元 Series」に安全に変換する。
    （yfinanceの戻りが (N,1) ndarray になることがあるため）
    """
    if "Close" not in df.columns:
        raise ValueError("Close column not found.")
    close = df["Close"]
    # DataFrame->Series の場合や、ndarray 形状を吸収
    if isinstance(close, pd.DataFrame):
        close = close.squeeze("columns")
    if not isinstance(close, pd.Series):
        close = pd.Series(close, index=df.index)
    # 数値化＋欠損除去
    close = pd.to_numeric(close, errors="coerce").dropna()
    return close

def fetch_prev_close(ticker: str) -> float:
    d = yf.download(ticker, period="10d", interval="1d", auto_adjust=False, progress=False)
    if d.empty:
        raise RuntimeError(f"[WARN] prev close empty for {ticker}")
    s = ensure_series_1dClose(d)
    # 前日終値（直近の1つ前）
    if len(s) < 2:
        # 1本しか無い場合は最後（今日）を前日としてみなさないように safety
        return float(s.iloc[-1])
    return float(s.iloc[-2])

def fetch_intraday_series(ticker: str) -> pd.Series:
    d = yf.download(
        ticker, period=INTRA_PERIOD, interval=INTRA_INTERVAL,
        auto_adjust=False, progress=False
    )
    if d.empty:
        raise RuntimeError(f"[WARN] intraday empty for {ticker}")
    s = ensure_series_1dClose(d)
    # 同一日のみ抽出（最後に近い営業日）: index が tz-aware の場合も想定
    # 当日JSTの日付でフィルタ
    last_day = pd.to_datetime(s.index[-1]).astimezone(BASE_TZ).date()
    s = s[pd.to_datetime(s.index).tz_convert(BASE_TZ).date == last_day]
    if s.empty:
        raise RuntimeError(f"[WARN] intraday filtered empty for {ticker}")
    return s

# ---------- 指数構築 ----------
def build_equal_weight_index(tickers: List[str]) -> pd.DataFrame:
    rows = []
    for t in tickers:
        try:
            print(f"[INFO] Fetching {t} ...")
            prev = fetch_prev_close(t)
            intraday = fetch_intraday_series(t)

            pct = (intraday / prev - 1.0) * 100.0
            pct = pct.rename(t)
            rows.append(pct)

        except Exception as e:
            print(f"[WARN] skip {t}  # {e}")

    if not rows:
        raise RuntimeError("取得できた日中データが0でした。ティッカーを見直してください。")

    df = pd.concat(rows, axis=1).sort_index()
    # 等ウェイト
    df["R_BANK9"] = df.mean(axis=1, skipna=True)
    return df

# ---------- 可視化 ----------
def pick_line_color(series: pd.Series) -> str:
    """
    終端がプラスなら青緑、マイナスなら赤
    """
    if len(series) == 0:
        return "#00e5d7"
    return "#00e5d7" if float(series.iloc[-1]) >= 0 else "#ff4d4d"

def plot_index(df: pd.DataFrame) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    series = df["R_BANK9"]
    c = pick_line_color(series)

    plt.close("all")
    fig = plt.figure(figsize=(16, 9), dpi=160)
    ax = fig.add_subplot(111)

    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    for spine in ax.spines.values():
        spine.set_color("#444444")

    ax.plot(series.index, series.values, color=c, linewidth=3.0, label="R-BANK9")
    ax.axhline(0, color="#666666", linewidth=1.0)
    ax.tick_params(colors="white")
    ax.set_title(f"R-BANK9 Intraday Snapshot ({jst_now().strftime('%Y/%m/%d %H:%M')})",
                 color="white", fontsize=22, pad=12)
    ax.set_xlabel("Time", color="white")
    ax.set_ylabel("Change vs Prev Close (%)", color="white")
    ax.legend(facecolor="black", edgecolor="#444444", labelcolor="white", loc="upper left")
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
            f"※ 構成9銘柄の等ウェイト\n"
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
