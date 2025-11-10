# -*- coding: utf-8 -*-
"""
R-BANK9 intraday index snapshot
等ウェイト / 前日終値比（%）で1日チャートを描画（黒背景・SNS向け）
欠損バーの影響で終盤に指数が暴れる問題を、ユニオン時系列 + 前値保有 + 被覆率フィルタで改善
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
TICKER_FILE = "docs/tickers_rbank9.txt"   # 5830.T などを1行1ティッカー

IMG_PATH = os.path.join(OUT_DIR, "rbank9_intraday.png")
CSV_PATH = os.path.join(OUT_DIR, "rbank9_intraday.csv")
POST_PATH = os.path.join(OUT_DIR, "rbank9_post_intraday.txt")

# JP は 1m が不安定なことがあるので 5m で安定運用
INTRA_PERIOD = "7d"
INTRA_INTERVAL = "5m"

# index 構築の堅牢化パラメータ
COVERAGE_MIN_FRAC = 0.70  # この割合未満しか値がない時刻は指数から除外
MARKET_OPEN_JST = dtime(9, 0)
MARKET_CLOSE_JST = dtime(15, 0)

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

def _to_series_1d(close_like: pd.DataFrame | pd.Series, index) -> pd.Series:
    """
    yfinance の Close が (N,), (N,1), (N,k) など何で来ても
    1 次元 Series[float] に正規化する。
    - 数値化（coerce）
    - 複数列ある場合：有効データ点数が最大の列を採用
    """
    if isinstance(close_like, pd.Series):
        ser = pd.to_numeric(close_like, errors="coerce").dropna()
        ser.index = index
        return ser

    df = close_like.apply(pd.to_numeric, errors="coerce")
    mask = df.notna().any(axis=0)  # ← axis を明示
    df = df.loc[:, mask]

    if df.shape[1] == 0:
        raise ValueError("no numeric close column")

    if df.shape[1] == 1:
        ser = df.iloc[:, 0]
    else:
        best_col = df.count(axis=0).idxmax()
        ser = df[best_col]

    ser = ser.astype(float)
    ser.index = index
    ser = ser.dropna()
    return ser

def ensure_series_1dClose(df: pd.DataFrame) -> pd.Series:
    if "Close" not in df.columns:
        raise ValueError("Close column not found")
    close = df["Close"]
    return _to_series_1d(close, df.index)

def _to_jst_indexed(s: pd.Series) -> pd.Series:
    """index を JST に正規化し、時刻のみを残して返す（タイムゾーン付）"""
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
    if len(s) >= 2:
        return float(s.iloc[-2])  # 前日終値
    return float(s.iloc[-1])

def fetch_intraday_series(ticker: str) -> pd.Series:
    d = yf.download(ticker, period=INTRA_PERIOD, interval=INTRA_INTERVAL,
                    auto_adjust=False, progress=False)
    if d.empty:
        raise RuntimeError(f"[WARN] intraday empty for {ticker}")
    s = ensure_series_1dClose(d)
    s = _to_jst_indexed(s)

    # 当日(JST)だけ抽出
    last_day = s.index[-1].date()
    s = s[s.index.date == last_day]

    # 市場時間に限定
    s = s[(s.index.time >= MARKET_OPEN_JST) & (s.index.time <= MARKET_CLOSE_JST)]

    if s.empty:
        raise RuntimeError(f"[WARN] intraday filtered empty for {ticker}")
    return s

# ---------- 指数構築（ユニオン + 前値保有 + 被覆率） ----------
def build_equal_weight_index(tickers: List[str]) -> pd.DataFrame:
    pct_map: Dict[str, pd.Series] = {}
    prev_map: Dict[str, float] = {}

    for t in tickers:
        try:
            print(f"[INFO] Fetching {t} ...")
            prev = fetch_prev_close(t)
            cur = fetch_intraday_series(t)           # JST, 当日, 取引時間内
            pct = (cur / prev - 1.0) * 100.0         # 前日終値比(%)
            pct_map[t] = pct.rename(t)
            prev_map[t] = prev
        except Exception as e:
            print(f"[WARN] skip {t}  # {e}")

    if not pct_map:
        raise RuntimeError("取得できた日中データが0でした。ティッカーを見直してください。")

    # すべての時刻のユニオンに合わせて列を結合
    df = pd.concat(pct_map.values(), axis=1).sort_index()

    # 前値保有（forward-fill）：各銘柄の最初の観測以降に適用
    df = df.ffill()

    # 被覆率フィルタ：値が入っている列の割合が一定未満の行は除外
    min_cols = max(1, int(len(pct_map) * COVERAGE_MIN_FRAC))
    coverage = df.count(axis=1)
    df = df[coverage >= min_cols]

    if df.empty:
        raise RuntimeError("被覆率フィルタ後にデータが空になりました。しきい値を見直してください。")

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
    ax.plot(series.index, series.values, color=c, linewidth=3.0, label="R-BANK9")
    ax.axhline(0, color="#666666", linewidth=1.0)
    ax.tick_params(colors="white")
    ax.set_title(
        f"R-BANK9 Intraday Snapshot ({jst_now().strftime('%Y/%m/%d %H:%M')})",
        color="white", fontsize=22, pad=12
    )
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
            f"※ 構成9銘柄の等ウェイト（欠損は直前値保有・被覆率{int(COVERAGE_MIN_FRAC*100)}%未満の時刻は除外）\n"
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
