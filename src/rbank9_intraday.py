# -*- coding: utf-8 -*-
"""
R-BANK9 Intraday Snapshot
等金額（等ウェイト）で前日終値比（%）を5〜60分足で集計し、1日のラインを出力。
- 入力: docs/tickers_rbank9.txt
- 出力: docs/outputs/rbank9_intraday.png / rbank9_intraday.csv / rbank9_post_intraday.txt
"""

import os
from datetime import datetime, timezone, timedelta
from typing import List

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ===== パス =====
TICKER_FILE = "docs/tickers_rbank9.txt"
OUT_DIR = "docs/outputs"
IMG_PATH = os.path.join(OUT_DIR, "rbank9_intraday.png")
CSV_PATH = os.path.join(OUT_DIR, "rbank9_intraday.csv")
POST_PATH = os.path.join(OUT_DIR, "rbank9_post_intraday.txt")

os.makedirs(OUT_DIR, exist_ok=True)

# ===== ユーティリティ =====
def jst_now():
    return datetime.now(timezone(timedelta(hours=9)))

def read_tickers(path: str) -> List[str]:
    tickers: List[str] = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} が見つかりません。ティッカー一覧を用意してください。")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            tickers.append(s)
    if not tickers:
        raise RuntimeError("ティッカーが1つも読み込めませんでした。docs/tickers_rbank9.txt を確認してください。")
    return tickers

def fetch_prev_close(ticker: str) -> float:
    """
    直近営業日の終値を取得（d日足から）。取りこぼし時は例外。
    """
    df = yf.download(ticker, period="10d", interval="1d", progress=False, auto_adjust=False)
    if df is None or df.empty or "Close" not in df:
        raise RuntimeError(f"prev close not found: {ticker}")
    s = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if s.empty:
        raise RuntimeError(f"prev close empty after coerce: {ticker}")
    return float(s.iloc[-1])

def fetch_intraday_series(ticker: str) -> pd.Series:
    """
    当日の日中足（5m→15m→60mフォールバック）を終値の前日比(%)で返す。
    """
    for iv in ["5m", "15m", "60m"]:
        df = yf.download(ticker, period="1d", interval=iv, progress=False, auto_adjust=False)
        if df is not None and not df.empty and "Close" in df.columns:
            closes = pd.to_numeric(df["Close"], errors="coerce").dropna()
            if closes.empty:
                continue
            try:
                prev = fetch_prev_close(ticker)
            except Exception as e:
                print(f"[WARN] prev close fetch failed for {ticker}: {e}")
                continue
            pct = closes / prev - 1.0
            pct.name = ticker
            return pct
    raise RuntimeError("no intraday data")

def safe_mean(df: pd.DataFrame) -> pd.Series:
    """行方向の平均（NaNは無視）。全列NaNの行はNaN。"""
    return df.mean(axis=1, skipna=True)

# ===== メイン =====
def main():
    print("[INFO] Building R-BANK9 intraday index ...")
    tickers = read_tickers(TICKER_FILE)

    series_dict = {}
    failed = []

    for t in tickers:
        try:
            s = fetch_intraday_series(t)
            series_dict[t] = s
            print(f"[INFO] ok: {t}, points={len(s)}")
        except Exception as e:
            failed.append((t, str(e)))
            print(f"[WARN] skip {t}: {e}")

    if not series_dict:
        raise RuntimeError("取得できた日中データが0でした。ティッカーを見直してください。")

    df = pd.DataFrame(series_dict)

    # 等ウェイトでバスケット（%）
    basket = safe_mean(df) * 100.0
    basket.name = "R-BANK9"

    # CSV保存（時刻・各銘柄%・バスケット%）
    out = df.mul(100.0).copy()
    out["R-BANK9"] = basket
    out.index.name = "Time"
    out.to_csv(CSV_PATH, float_format="%.6f")
    print(f"[INFO] saved csv -> {CSV_PATH}")

    # 描画：終値が前日比プラスなら青緑、マイナスなら赤
    last_pct = basket.dropna().iloc[-1] if not basket.dropna().empty else 0.0
    color = "#00E5D4" if last_pct >= 0 else "#FF554D"

    plt.close("all")
    fig = plt.figure(figsize=(16, 9), dpi=160)
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    ax.plot(basket.index, basket.values, color=color, linewidth=3.0, label="R-BANK9")
    ax.axhline(0, color="#666666", linewidth=1.0)

    # 軸・スパイン色
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#444444")

    ax.set_title(f"R-BANK9 Intraday Snapshot ({jst_now().strftime('%Y/%m/%d %H:%M')})",
                 color="white", fontsize=22, pad=12)
    ax.set_xlabel("Time", color="white")
    ax.set_ylabel("Change vs Prev Close (%)", color="white")
    ax.legend(facecolor="black", edgecolor="#444444", labelcolor="white", loc="upper left")

    fig.tight_layout()
    plt.savefig(IMG_PATH, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] saved chart -> {IMG_PATH}")

    # 投稿テキスト
    sign = "🔺" if last_pct >= 0 else "🔻"
    with open(POST_PATH, "w", encoding="utf-8") as f:
        f.write(
            f"{sign} R-BANK9 日中スナップショット（{jst_now().strftime('%Y/%m/%d %H:%M')}）\n"
            f"{last_pct:+.2f}%（前日終値比）\n"
            f"※ 構成9銘柄の等ウェイト\n"
            f"#地方銀行 #R_BANK9 #日本株\n"
        )
    print(f"[INFO] saved post -> {POST_PATH}")

    if failed:
        print("\n[WARN] 以下は取得失敗（参考）：")
        for t, msg in failed:
            print(f"  - {t}: {msg}")

    print("[INFO] intraday outputs:")
    print(os.path.abspath(IMG_PATH))
    print(os.path.abspath(CSV_PATH))
    print(os.path.abspath(POST_PATH))


if __name__ == "__main__":
    main()
