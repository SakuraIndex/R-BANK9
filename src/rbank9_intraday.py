# -*- coding: utf-8 -*-
"""
R-BANK9 intraday index snapshot
等ウェイト / 前日終値比（%）で1日チャートを描画（黒背景・SNS向け）

不自然なスパイク対策：
- 同日の5分グリッド（JST）に各銘柄を整列して ffill
- 全銘柄の最終タイムスタンプの最小値までトリム（共通区間のみ採用）
- 騰落率は ±20% でクリップ（yfinanceの瞬間異常値対策）
"""

import os
from typing import List, Dict
from datetime import datetime, timezone, timedelta

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

# クリップ（安全弁）
PCT_CLIP_LOW = -20.0
PCT_CLIP_HIGH = 20.0

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
    - すべて数値化（coerce）
    - 複数列ある場合：有効データ点数が最大の列を採用
    """
    if isinstance(close_like, pd.Series):
        ser = pd.to_numeric(close_like, errors="coerce").dropna()
        return ser

    df = close_like.apply(pd.to_numeric, errors="coerce")
    mask = df.notna().any(axis=0)
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

def fetch_prev_close(ticker: str) -> float:
    d = yf.download(ticker, period="10d", interval="1d",
                    auto_adjust=False, progress=False)
    if d.empty:
        raise RuntimeError(f"[WARN] prev close empty for {ticker}")
    s = ensure_series_1dClose(d)
    if len(s) >= 2:
        return float(s.iloc[-2])
    return float(s.iloc[-1])

def fetch_intraday_series(ticker: str) -> pd.Series:
    d = yf.download(ticker, period=INTRA_PERIOD, interval=INTRA_INTERVAL,
                    auto_adjust=False, progress=False)
    if d.empty:
        raise RuntimeError(f"[WARN] intraday empty for {ticker}")
    s = ensure_series_1dClose(d)

    # 当日(JST)だけ抽出
    idx = pd.to_datetime(s.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert(BASE_TZ)
    last_day = idx[-1].date()
    s = pd.Series(s.values, index=idx)
    s = s[(s.index.date == last_day)]
    if s.empty:
        raise RuntimeError(f"[WARN] intraday filtered empty for {ticker}")
    return s

# ---------- 指数構築 ----------
def build_equal_weight_index(tickers: List[str]) -> pd.DataFrame:
    # 個別の前日終値 & 当日系列を取得
    indiv_pct: Dict[str, pd.Series] = {}
    last_times: List[pd.Timestamp] = []

    for t in tickers:
        try:
            print(f"[INFO] Fetching {t} ...")
            prev = fetch_prev_close(t)
            intraday = fetch_intraday_series(t)
            pct = (intraday / prev - 1.0) * 100.0
            # 安全弁クリップ（明らかな誤値を抑制）
            pct = pct.clip(lower=PCT_CLIP_LOW, upper=PCT_CLIP_HIGH)
            indiv_pct[t] = pct.rename(t)
            last_times.append(pct.index.max())
        except Exception as e:
            print(f"[WARN] skip {t}  # {e}")

    if not indiv_pct:
        raise RuntimeError("取得できた日中データが0でした。ティッカーを見直してください。")

    # 全銘柄の最終タイムスタンプの最小値（＝共通に揃う終了時刻）
    common_end = min(last_times)

    # その日の共通グリッド（5分足, JST）
    # 始点は各シリーズの最小時刻の最大値にしてもよいが、
    # 可視化のため最小～common_endで十分。ffillで整う。
    start_time = min(s.index.min() for s in indiv_pct.values())
    grid = pd.date_range(start=start_time, end=common_end, freq=INTRA_INTERVAL, tz=BASE_TZ)

    # グリッドに合わせて reindex + ffill で列を作る
    aligned = []
    for t, ser in indiv_pct.items():
        s2 = ser.reindex(grid).ffill()
        aligned.append(s2.rename(t))

    df = pd.concat(aligned, axis=1)
    # 等ウェイト平均
    df["R_BANK9"] = df.mean(axis=1, skipna=True)

    # CSV用に保存しやすい形で返す
    df = df.sort_index()
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
