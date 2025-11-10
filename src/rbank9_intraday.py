# -*- coding: utf-8 -*-
"""
R-BANK9 intraday index snapshot
等ウェイト / 前日終値比（%）で1日チャートを描画（黒背景・SNS向け）
欠損アラインの不良で右端が跳ね上がる事象を修正：
- 当日JSTセッションの5分グリッドへ再インデックス＋前方埋め
- 有効銘柄が閾値未満の行は除外
"""

import os
from typing import List
from datetime import datetime, timezone, timedelta, date

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ---------- 設定 ----------
BASE_TZ = timezone(timedelta(hours=9))  # JST
OUT_DIR = "docs/outputs"
TICKER_FILE = "docs/tickers_rbank9.txt"   # 5830.T などを1行1ティッカー

IMG_PATH = os.path.join(OUT_DIR, "rbank9_intraday.png")
CSV_PATH = os.path.join(OUT_DIR, "rbank9_intraday.csv")
POST_PATH = os.path.join(OUT_DIR, "rbank9_post_intraday.txt")

# JP は 1m が不安定なことがあるので 5m で安定運用
INTRA_PERIOD = "7d"
INTRA_INTERVAL = "5m"

# 行を採用する最小の有効銘柄数（例：9銘柄中 7 以上）
MIN_COUNT_RATIO = 0.7

# ---------- ユーティリティ ----------
def jst_now() -> datetime:
    return datetime.now(BASE_TZ)

def today_jst() -> date:
    return jst_now().date()

def session_grid(d: date) -> pd.DatetimeIndex:
    """当日の 09:00〜15:25 JST の5分グリッド（終値刻み）"""
    start = datetime(d.year, d.month, d.day, 9, 0, tzinfo=BASE_TZ)
    end   = datetime(d.year, d.month, d.day, 15, 25, tzinfo=BASE_TZ)
    return pd.date_range(start, end, freq="5min", tz=BASE_TZ)

def load_tickers(path: str) -> List[str]:
    xs: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            xs.append(s)
    return xs

def _to_series_1d(close_like: pd.DataFrame | pd.Series, index) -> pd.Series:
    """yfinance の Close が (N,), (N,1), (N,k) など何で来ても 1D Series[float] に正規化"""
    if isinstance(close_like, pd.Series):
        ser = pd.to_numeric(close_like, errors="coerce").dropna()
        ser.index = index
        return ser

    df = close_like.apply(pd.to_numeric, errors="coerce")
    mask = df.notna().any(axis=0)
    df = df.loc[:, mask]
    if df.shape[1] == 0:
        raise ValueError("no numeric close column")
    ser = df.iloc[:, 0] if df.shape[1] == 1 else df[df.count(axis=0).idxmax()]
    ser = pd.to_numeric(ser, errors="coerce").dropna()
    ser.index = index
    return ser

def ensure_series_1dClose(df: pd.DataFrame) -> pd.Series:
    if "Close" not in df.columns:
        raise ValueError("Close column not found")
    close = df["Close"]
    return _to_series_1d(close, df.index)

def fetch_prev_close(ticker: str) -> float:
    d = yf.download(ticker, period="10d", interval="1d",
                    auto_adjust=False, progress=False)
    if d.empty:
        raise RuntimeError(f"[WARN] prev close empty for {ticker}")
    s = ensure_series_1dClose(d)
    # 前日終値（直近 1 本前）
    if len(s) >= 2:
        return float(s.iloc[-2])
    return float(s.iloc[-1])

def fetch_intraday_series(ticker: str) -> pd.Series:
    d = yf.download(ticker, period=INTRA_PERIOD, interval=INTRA_INTERVAL,
                    auto_adjust=False, progress=False)
    if d.empty:
        raise RuntimeError(f"[WARN] intraday empty for {ticker}")
    s = ensure_series_1dClose(d)

    # UTC -> JST
    idx = pd.to_datetime(s.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert(BASE_TZ)

    s.index = idx

    # 当日(JST)だけ抽出
    d0 = today_jst()
    s = s[(s.index.date == d0)]
    if s.empty:
        raise RuntimeError(f"[WARN] intraday filtered empty for {ticker}")

    return s

# ---------- 指数構築 ----------
def build_equal_weight_index(tickers: List[str]) -> pd.DataFrame:
    d0 = today_jst()
    grid = session_grid(d0)

    rows = []
    for t in tickers:
        try:
            print(f"[INFO] Fetching {t} ...")
            prev = fetch_prev_close(t)
            intraday = fetch_intraday_series(t)
            pct = (intraday / prev - 1.0) * 100.0

            # ★ 当日セッションの5分グリッドに再インデックス＋当日内のみ前方埋め
            pct = pct.reindex(grid).ffill()
            rows.append(pct.rename(t))
        except Exception as e:
            print(f"[WARN] skip {t}  # {e}")

    if not rows:
        raise RuntimeError("取得できた日中データが0でした。ティッカーを見直してください。")

    df = pd.concat(rows, axis=1)  # すべて grid 上にあるので列結合でOK

    # 行の有効銘柄数が少なすぎる時間は除外（右端スパイク対策）
    min_count = max(1, int(len(tickers) * MIN_COUNT_RATIO))
    valid_cnt = df.notna().sum(axis=1)
    mean_series = df.mean(axis=1, skipna=True)
    mean_series[valid_cnt < min_count] = pd.NA
    mean_series = mean_series.dropna()

    out = df.copy()
    out["R_BANK9"] = mean_series

    return out

# ---------- 可視化 ----------
def pick_line_color(series: pd.Series) -> str:
    return "#00e5d7" if len(series) and float(series.iloc[-1]) >= 0 else "#ff4d4d"

def plot_index(df: pd.DataFrame) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    series = df["R_BANK9"].dropna()
    if series.empty:
        raise RuntimeError("plot_index: series is empty")

    c = pick_line_color(series)

    plt.close("all")
    fig = plt.figure(figsize=(16, 9), dpi=160)
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    for sp in ax.spines.values():
        sp.set_color("#444444")
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
    df.to_csv(CSV_PATH, encoding="utf-8", index_label="datetime_jst")

def save_post_text(df: pd.DataFrame, tickers: List[str]) -> None:
    series = df["R_BANK9"].dropna()
    last = float(series.iloc[-1]) if len(series) else 0.0
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
