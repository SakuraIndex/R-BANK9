# -*- coding: utf-8 -*-
"""
R-BANK9 intraday index snapshot
等ウェイト / 前日終値比（%）で1日チャートを描画（黒背景・SNS向け）
右端スパイク対策：
  - 5分グリッドに整列し各銘柄の価格は前回値でロールフォワード
  - 時刻ごとの有効銘柄数が閾値未満(既定:70%)の行は採用しない
"""

from __future__ import annotations

import os
from typing import List, Tuple
from datetime import datetime, date, time, timedelta, timezone

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ---------- 設定 ----------
JST = timezone(timedelta(hours=9))
OUT_DIR = "docs/outputs"
TICKER_FILE = "docs/tickers_rbank9.txt"   # 5830.T など1行1ティッカー

IMG_PATH = os.path.join(OUT_DIR, "rbank9_intraday.png")
CSV_PATH = os.path.join(OUT_DIR, "rbank9_intraday.csv")
POST_PATH = os.path.join(OUT_DIR, "rbank9_post_intraday.txt")
STATS_PATH = os.path.join(OUT_DIR, "rbank9_stats.json")

# 取得設定
INTRA_PERIOD = "7d"     # 安定運用のため 5m × 数日
INTRA_INTERVAL = "5m"

# セッション（日本株）
SESSION_START = time(9, 0)
SESSION_END   = time(15, 30)

# カバレッジ閾値（有効銘柄率。例: 0.7=70%未満は棄却）
MIN_COVERAGE = 0.70

# ---------- ユーティリティ ----------
def jst_now() -> datetime:
    return datetime.now(JST)

def today_jst() -> date:
    return jst_now().date()

def load_tickers(path: str) -> List[str]:
    xs: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            xs.append(s)
    if not xs:
        raise RuntimeError("ティッカーが0件です。docs/tickers_rbank9.txt を確認してください。")
    return xs

def ensure_1d_close(obj: pd.DataFrame | pd.Series) -> pd.Series:
    """yfinanceの Close を安全に 1D Series[float] へ正規化"""
    if isinstance(obj, pd.Series):
        s = pd.to_numeric(obj, errors="coerce")
        return s

    df = obj.apply(pd.to_numeric, errors="coerce")
    mask = df.notna().any(axis=0)
    df = df.loc[:, mask]
    if df.shape[1] == 0:
        raise ValueError("Close列が取得できませんでした。")

    if df.shape[1] == 1:
        s = df.iloc[:, 0]
    else:
        # 有効データ点が最大の列を採用
        best = df.count(axis=0).idxmax()
        s = df[best]
    return s.astype(float)

def fetch_prev_close(ticker: str) -> float:
    d = yf.download(ticker, period="10d", interval="1d",
                    auto_adjust=False, progress=False)
    if d.empty:
        raise RuntimeError(f"prev close empty: {ticker}")
    s = ensure_1d_close(d["Close"] if "Close" in d.columns else d)
    # 前日終値（直近の一本前）。1本しか無い場合は直近値を使う
    return float(s.iloc[-2] if len(s) >= 2 else s.iloc[-1])

def fetch_intraday_close(ticker: str) -> pd.Series:
    d = yf.download(ticker, period=INTRA_PERIOD, interval=INTRA_INTERVAL,
                    auto_adjust=False, progress=False)
    if d.empty:
        raise RuntimeError(f"intraday empty: {ticker}")
    s = ensure_1d_close(d["Close"] if "Close" in d.columns else d)

    # インデックスをJSTに
    idx = pd.to_datetime(s.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert(JST)
    s.index = idx

    # 今日(JST)のみ
    d0 = today_jst()
    s = s[(s.index.date == d0)]
    if s.empty:
        raise RuntimeError(f"intraday today empty: {ticker}")
    return s

def make_session_grid(d: date) -> pd.DatetimeIndex:
    start = datetime.combine(d, SESSION_START, tzinfo=JST)
    end   = datetime.combine(d, SESSION_END, tzinfo=JST)
    # 5分足グリッド（終端を含める）
    return pd.date_range(start, end, freq="5min", tz=JST)

# ---------- 指数構築 ----------
def build_equal_weight_index(tickers: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    grid = make_session_grid(today_jst())

    # 各ティッカーの前日終値と日中終値Seriesを取得→グリッドに合わせてFFILL
    cols = {}
    prev_map = {}
    for t in tickers:
        try:
            prev = fetch_prev_close(t)
            s = fetch_intraday_close(t)
            # グリッドに再インデックス（約定のない足は直前値をFFILL）
            s = s.reindex(grid).ffill()
            cols[t] = s
            prev_map[t] = prev
        except Exception as e:
            print(f"[WARN] skip {t}  # {e}")

    if not cols:
        raise RuntimeError("取得できた銘柄が0です。")

    price_df = pd.DataFrame(cols)  # index=grid
    # 価格→前日終値比（%）
    pct_df = pd.DataFrame({
        t: (price_df[t] / prev_map[t] - 1.0) * 100.0
        for t in cols.keys()
    }, index=grid)

    # カバレッジ（非NaN銘柄数）
    valid_count = pct_df.notna().sum(axis=1)
    min_need = max(1, int(len(cols) * MIN_COVERAGE + 0.0001))
    mask_ok = valid_count >= min_need

    if not mask_ok.any():
        raise RuntimeError("カバレッジ条件を満たす時刻がありません。")

    # 「最後に条件を満たした時刻」までで打ち切り（右端スパイク除去）
    last_good_ts = mask_ok[mask_ok].index[-1]
    pct_df = pct_df.loc[:last_good_ts]

    # 等ウェイト平均
    pct_df["R_BANK9"] = pct_df.mean(axis=1, skipna=True)

    return pct_df, pct_df["R_BANK9"]

# ---------- 可視化 ----------
def pick_line_color(series: pd.Series) -> str:
    return "#00e5d7" if len(series) and float(series.iloc[-1]) >= 0 else "#ff4d4d"

def plot_series(series: pd.Series) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    c = pick_line_color(series)

    plt.close("all")
    fig = plt.figure(figsize=(16, 9), dpi=160)
    ax = fig.add_subplot(111)

    # 黒背景＋ティールの既存デザイン
    fig.patch.set_facecolor("#000000")
    ax.set_facecolor("#000000")
    for sp in ax.spines.values():
        sp.set_color("#444444")

    ax.plot(series.index, series.values, color=c, linewidth=3.0)
    ax.axhline(0, color="#666666", linewidth=1.0)

    ax.tick_params(colors="#dddddd")
    ax.set_title(f"R-BANK9 Intraday Snapshot ({jst_now().strftime('%Y/%m/%d %H:%M JST')})",
                 color="#ffffff", fontsize=22, pad=12)
    ax.set_xlabel("Time", color="#dddddd")
    ax.set_ylabel("Change vs Prev Close (%)", color="#dddddd")

    fig.tight_layout()
    plt.savefig(IMG_PATH, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

# ---------- 出力 ----------
def save_csv(pct_df: pd.DataFrame) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    out = pct_df.copy()
    out.index.name = "datetime_jst"
    out.to_csv(CSV_PATH, encoding="utf-8")

def save_post_text(series: pd.Series, tickers: List[str]) -> None:
    last = float(series.iloc[-1])
    sign = "🔺" if last >= 0 else "🔻"
    with open(POST_PATH, "w", encoding="utf-8") as f:
        f.write(
            f"{sign} R-BANK9 日中スナップショット（{jst_now().strftime('%Y/%m/%d %H:%M JST')}）\n"
            f"{last:+.2f}%（前日終値比）\n"
            f"※ 構成{len(tickers)}銘柄の等ウェイト\n"
            f"#地方銀行 #R_BANK9 #日本株\n"
        )

def save_stats(series: pd.Series) -> None:
    import json
    os.makedirs(OUT_DIR, exist_ok=True)
    stats = {
        "index_key": "R_BANK9",
        "label": "R-BANK9",
        "pct_intraday": float(series.iloc[-1]) if len(series) else 0.0,
        "basis": "prev_close",
        "session": {"start": "09:00", "end": "15:30", "anchor": "09:00"},
        "updated_at": jst_now().isoformat(),
    }
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

# ---------- メイン ----------
def main():
    tickers = load_tickers(TICKER_FILE)
    print("[INFO] Building R_BANK9 intraday index ...")

    pct_df, series = build_equal_weight_index(tickers)

    # 出力
    plot_series(series)
    save_csv(pct_df)
    save_post_text(series, tickers)
    save_stats(series)

    print("[INFO] done.")

if __name__ == "__main__":
    main()
