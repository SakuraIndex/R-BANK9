# -*- coding: utf-8 -*-
"""
R-BANK9 intraday index snapshot (robust)
等ウェイト / 前日終値比（%）で1日チャートを描画（黒背景・SNS向け）
終盤だけデータが残る銘柄の影響を除外して安定化。
"""

import os
from typing import List
from math import ceil
from datetime import datetime, timezone, timedelta
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

MIN_COVERAGE = 0.7       # 有効銘柄が全体の70%未満なら指数更新しない
OUTLIER_ABS_PCT = 15.0   # ±15%以上のバーは除外


# ---------- ユーティリティ ----------
def jst_now() -> datetime:
    return datetime.now(BASE_TZ)

def load_tickers(path: str) -> List[str]:
    tickers = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if s and not s.startswith("#"):
                tickers.append(s)
    return tickers


def ensure_series_1dClose(df: pd.DataFrame) -> pd.Series:
    """DataFrame → 1次元Closeシリーズ"""
    if "Close" not in df.columns:
        raise ValueError("Close column not found")
    s = pd.to_numeric(df["Close"], errors="coerce")
    s = s.dropna()
    return s


def fetch_prev_close(ticker: str) -> float:
    d = yf.download(ticker, period="10d", interval="1d",
                    auto_adjust=False, progress=False)
    if d.empty:
        raise RuntimeError(f"[WARN] prev close empty for {ticker}")
    s = ensure_series_1dClose(d)
    return float(s.iloc[-2]) if len(s) >= 2 else float(s.iloc[-1])


def fetch_intraday_series(ticker: str) -> pd.Series:
    """当日(JST)分の intraday Close series"""
    d = yf.download(ticker, period=INTRA_PERIOD, interval=INTRA_INTERVAL,
                    auto_adjust=False, progress=False)
    if d.empty:
        raise RuntimeError(f"[WARN] intraday empty for {ticker}")
    s = ensure_series_1dClose(d)

    idx = pd.to_datetime(s.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert(BASE_TZ)
    today = idx[-1].date()
    s = s[idx.date == today]
    s.index = idx[idx.date == today]
    return s


# ---------- 指数構築 ----------
def build_equal_weight_index(tickers: List[str]) -> pd.DataFrame:
    series_list = []
    for t in tickers:
        try:
            prev = fetch_prev_close(t)
            intraday = fetch_intraday_series(t)
            pct = (intraday / prev - 1.0) * 100.0
            pct = pct.mask(pct.abs() > OUTLIER_ABS_PCT)
            series_list.append(pct.rename(t))
            print(f"[INFO] fetched {t} ({len(pct)} pts)")
        except Exception as e:
            print(f"[WARN] skip {t}: {e}")

    if not series_list:
        raise RuntimeError("No intraday data fetched")

    # すべての時刻を共通インデックスに統一
    all_times = pd.Index(sorted(set().union(*[s.index for s in series_list])))
    df = pd.DataFrame(index=all_times)
    for s in series_list:
        df[s.name] = s.reindex(all_times)

    # 時系列を時刻順に並べ、欠損を前方補間
    df = df.sort_index().ffill()

    # 被覆率でマスク（指数更新停止処理）
    min_required = ceil(len(df.columns) * MIN_COVERAGE)
    valid_counts = df.notna().sum(axis=1)
    masked_df = df.where(valid_counts >= min_required)

    # 等ウェイト平均（有効列のみ）
    index_series = masked_df.mean(axis=1, skipna=True)

    # 被覆率閾値未満のNaNは前値キープ
    index_series = index_series.ffill()

    df["R_BANK9"] = index_series
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
    ax.set_title(
        f"R-BANK9 Intraday Snapshot ({jst_now().strftime('%Y/%m/%d %H:%M')})",
        color="white", fontsize=22, pad=12
    )
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
