# src/rbank9_intraday.py  修正完全版
# -*- coding: utf-8 -*-
"""
R-BANK9 intraday index snapshot
等ウェイト / 前日終値比（%）で1日チャートを描画（黒背景・SNS向け）
- 欠損が多いバーは集計しない（少数銘柄の値で平均が暴れるのを防止）
- 末尾の coverage 不足バーは切り落とし（終盤スパイク対策）
"""

from __future__ import annotations

import math
import os
from typing import Dict, List
from datetime import datetime, timezone, timedelta

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ---------- 設定 ----------
JST = timezone(timedelta(hours=9))
OUT_DIR = "docs/outputs"
TICKER_FILE = "docs/tickers_rbank9.txt"   # 5830.T などを1行1ティッカー

IMG_PATH = os.path.join(OUT_DIR, "rbank9_intraday.png")
CSV_PATH = os.path.join(OUT_DIR, "rbank9_intraday.csv")
POST_PATH = os.path.join(OUT_DIR, "rbank9_post_intraday.txt")
STATS_PATH = os.path.join(OUT_DIR, "rbank9_stats.json")

# 日本株は 1m が不安定なことがあるため 5m を使用
INTRA_PERIOD = "7d"
INTRA_INTERVAL = "5m"

# 取引時間（JST）
SESSION_START = "09:00"
SESSION_END = "15:30"

# 集計に必要な最小カバレッジ（割合）
MIN_COVERAGE_RATIO = 0.6  # 9銘柄なら 6 本以上そろった時だけ平均を採用

# ---------- ユーティリティ ----------
def jst_now() -> datetime:
    return datetime.now(JST)

def ensure_outdir() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

def load_tickers(path: str) -> List[str]:
    xs: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            xs.append(s)
    if not xs:
        raise RuntimeError("Tickers file is empty.")
    return xs

def _to_series_1d(close_like: pd.DataFrame | pd.Series, index) -> pd.Series:
    """
    yfinance の Close が (N,), (N,1), (N,k) など何で来ても 1 次元 Series[float] に正規化する。
    """
    if isinstance(close_like, pd.Series):
        ser = pd.to_numeric(close_like, errors="coerce").dropna()
        ser.index = index
        return ser

    df = close_like.apply(pd.to_numeric, errors="coerce")
    mask = df.notna().any(axis=0)  # ← any(axis=0) で完全欠損列を除外
    df = df.loc[:, mask]

    if df.shape[1] == 0:
        return pd.Series(dtype=float, index=index)

    if df.shape[1] == 1:
        ser = df.iloc[:, 0]
    else:
        best_col = df.count(axis=0).idxmax()
        ser = df[best_col]

    ser = ser.astype(float).dropna()
    ser.index = index
    return ser

def ensure_series_close(df: pd.DataFrame) -> pd.Series:
    if "Close" not in df.columns:
        # yfinance の返りで列名が MultiIndex のことがあるので、最後レベルが 'Close' の列を探す
        if isinstance(df.columns, pd.MultiIndex):
            close_cols = [c for c in df.columns if (isinstance(c, tuple) and c[-1] == "Close")]
            if not close_cols:
                raise ValueError("Close column not found")
            ser = _to_series_1d(df[close_cols], df.index)
            return ser
        raise ValueError("Close column not found")
    return _to_series_1d(df["Close"], df.index)

def fetch_prev_close(ticker: str) -> float:
    d = yf.download(ticker, period="10d", interval="1d",
                    auto_adjust=False, progress=False, threads=False)
    if d.empty:
        raise RuntimeError(f"[WARN] prev close empty for {ticker}")
    s = ensure_series_close(d)
    if len(s) >= 2:
        return float(s.iloc[-2])
    return float(s.iloc[-1])

def fetch_intraday_close(ticker: str) -> pd.Series:
    d = yf.download(ticker, period=INTRA_PERIOD, interval=INTRA_INTERVAL,
                    auto_adjust=False, progress=False, threads=False)
    if d.empty:
        raise RuntimeError(f"[WARN] intraday empty for {ticker}")
    s = ensure_series_close(d)
    # index -> JST
    idx = pd.to_datetime(s.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert(JST)
    s.index = idx

    # 当日だけ抽出
    today = jst_now().date()
    s = s[idx.date == today]
    if s.empty:
        raise RuntimeError(f"[WARN] intraday filtered empty for {ticker}")
    return s

def clip_session(s: pd.Series) -> pd.Series:
    """取引時間内にクリップ（JST）。"""
    if s.empty:
        return s
    start = pd.Timestamp(f"{s.index[0].date()} {SESSION_START}", tz=JST)
    end   = pd.Timestamp(f"{s.index[0].date()} {SESSION_END}", tz=JST)
    return s[(s.index >= start) & (s.index <= end)]

# ---------- 指数構築 ----------
def build_equal_weight_index(tickers: List[str]) -> pd.DataFrame:
    pct_map: Dict[str, pd.Series] = {}
    n = len(tickers)

    for t in tickers:
        try:
            prev = fetch_prev_close(t)
            intraday = fetch_intraday_close(t)
            intraday = clip_session(intraday)
            if intraday.empty:
                continue
            pct = (intraday / prev - 1.0) * 100.0
            pct_map[t] = pct.rename(t)
        except Exception as e:
            print(f"[WARN] skip {t}  # {e}")

    if not pct_map:
        raise RuntimeError("取得できた日中データが0でした。ティッカーとネットワークを確認してください。")

    # 横結合して時刻をそろえる（union）
    df = pd.concat(pct_map.values(), axis=1).sort_index()

    # coverage を計算（そのバーで実際に使えた銘柄数）
    cov = df.count(axis=1)
    df["coverage"] = cov

    # coverage が一定割合未満のバーは採用しない（= NaN）
    min_cov = max(1, math.ceil(n * MIN_COVERAGE_RATIO))
    mean_pct = df.drop(columns=["coverage"]).mean(axis=1, skipna=True)
    mean_pct[cov < min_cov] = float("nan")

    # 末尾の coverage 不足テイルを切り落とす（NaN を残さずスパイクを物理的に排除）
    if mean_pct.notna().any():
        last_valid = mean_pct.last_valid_index()
        mean_pct = mean_pct.loc[:last_valid]
        df = df.loc[mean_pct.index]  # CSV 側も同じ長さに合わせる

    out = pd.DataFrame({
        "R_BANK9": mean_pct,
        "coverage": df["coverage"].reindex(mean_pct.index)
    })
    return out

# ---------- 可視化 ----------
def pick_line_color(series: pd.Series) -> str:
    last = series.dropna().iloc[-1] if series.dropna().size else 0.0
    return "#00e5d7" if last >= 0 else "#ff4d4d"

def plot_index(series: pd.Series) -> None:
    ensure_outdir()
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
        f"R-BANK9 Intraday Snapshot (JST)\n{jst_now().strftime('%Y/%m/%d')}",
        color="white", fontsize=22, pad=12
    )
    ax.set_xlabel("Time", color="white")
    ax.set_ylabel("Change vs Prev Close (%)", color="white")
    ax.legend(facecolor="black", edgecolor="#444444", labelcolor="white", loc="upper left")

    fig.tight_layout()
    plt.savefig(IMG_PATH, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

# ---------- 出力 ----------
def save_csv(df: pd.DataFrame) -> None:
    ensure_outdir()
    df.to_csv(CSV_PATH, encoding="utf-8", index_label="datetime_jst")

def save_post_text(series: pd.Series, tickers: List[str]) -> None:
    ensure_outdir()
    last = float(series.dropna().iloc[-1]) if series.dropna().size else 0.0
    sign = "🔺" if last >= 0 else "🔻"
    with open(POST_PATH, "w", encoding="utf-8") as f:
        f.write(
            f"{sign} R-BANK9 日中スナップショット（{jst_now().strftime('%Y/%m/%d %H:%M')}）\n"
            f"{last:+.2f}%（前日終値比）\n"
            f"※ 構成9銘柄の等ウェイト / 欠損バーは除外\n"
            f"#地方銀行 #R_BANK9 #日本株\n"
        )

def save_stats(series: pd.Series) -> None:
    ensure_outdir()
    last = float(series.dropna().iloc[-1]) if series.dropna().size else 0.0
    stats = {
        "index_key": "R_BANK9",
        "label": "R-BANK9",
        "pct_intraday": last / 100.0 if False else last,  # サイト側が % 値をそのまま読む想定なら last を渡す
        "basis": "prev_close",
        "session": {"start": SESSION_START, "end": SESSION_END, "anchor": "09:00"},
        "updated_at": jst_now().isoformat(),
    }
    # 上の pct_intraday を「%」値のままにするか「比」にするかはサイト側仕様に合わせてください。
    # ここでは % 値（例: +2.34）をそのまま入れています。
    import json
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

# ---------- メイン ----------
def main():
    tickers = load_tickers(TICKER_FILE)

    df = build_equal_weight_index(tickers)
    save_csv(df)

    series = df["R_BANK9"]
    if series.dropna().empty:
        raise RuntimeError("有効な指数系列が生成できませんでした（coverage が閾値未満）。")

    plot_index(series)
    save_post_text(series, tickers)
    save_stats(series)
    print("[INFO] done.")

if __name__ == "__main__":
    main()
