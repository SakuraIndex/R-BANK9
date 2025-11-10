# -*- coding: utf-8 -*-
"""
R-BANK9 intraday index snapshot
等ウェイト / 前日終値比（%）で1日チャートを描画（黒背景・SNS向け）
- yfinance のローカルキャッシュを完全無効化して毎回新鮮なデータを取得
- ティッカー毎の間引き(レート制限/キャッシュ誤命中の回避)
- JST の当日セッション(09:00-15:30)のみを厳密抽出
- 欠損/列構造ゆらぎを吸収して 1D Series に正規化
"""

from __future__ import annotations

import os
import time
from typing import List
from datetime import datetime, timezone, timedelta, time as dtime

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ---------- yfinance キャッシュ完全無効化 ----------
try:
    # キャッシュを都度空にし、保存先も無効化
    yf.utils._history.empty_cache()
    yf._CACHE_DIR = None  # type: ignore[attr-defined]
except Exception:
    pass

# ---------- 設定 ----------
BASE_TZ = timezone(timedelta(hours=9))  # JST
OUT_DIR = "docs/outputs"
TICKER_FILE = os.path.join("docs", "tickers_rbank9.txt")  # 5830.T などを1行1ティッカー

IMG_PATH = os.path.join(OUT_DIR, "rbank9_intraday.png")
CSV_PATH = os.path.join(OUT_DIR, "rbank9_intraday.csv")
POST_PATH = os.path.join(OUT_DIR, "rbank9_post_intraday.txt")
STATS_PATH = os.path.join(OUT_DIR, "rbank9_stats.json")

# JP は 1m が不安定なことがあるので 5m で安定運用
INTRA_PERIOD = "7d"
INTRA_INTERVAL = "5m"

# 市場セッション（JST）
SESSION_START = dtime(hour=9, minute=0)
SESSION_END = dtime(hour=15, minute=30)

# Yahoo レート/キャッシュ誤命中回避のためのスリープ(秒)
FETCH_PAUSE_SEC = 0.6


# ---------- ユーティリティ ----------
def jst_now() -> datetime:
    return datetime.now(BASE_TZ)


def _ensure_outdir() -> None:
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
        raise RuntimeError(f"ティッカーが空です: {path}")
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
        ser.index = index
        return ser

    df = close_like.apply(pd.to_numeric, errors="coerce")
    mask = df.notna().any(axis=0)  # 1列でも値があれば採用
    df = df.loc[:, mask]

    if df.shape[1] == 0:
        raise ValueError("no numeric close column")

    if df.shape[1] == 1:
        ser = df.iloc[:, 0]
    else:
        best_col = df.count(axis=0).idxmax()
        ser = df[best_col]

    ser = ser.astype(float).dropna()
    ser.index = index
    return ser


def ensure_series_1d_close(df: pd.DataFrame) -> pd.Series:
    if "Close" not in df.columns:
        raise ValueError("Close column not found")
    close = df["Close"]
    return _to_series_1d(close, df.index)


def fetch_prev_close(ticker: str) -> float:
    d = yf.download(
        ticker,
        period="10d",
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if d is None or d.empty:
        raise RuntimeError(f"[WARN] prev close empty for {ticker}")
    s = ensure_series_1d_close(d)
    # 前日終値（直近 1 本前があればそれ、なければ最後）
    return float(s.iloc[-2] if len(s) >= 2 else s.iloc[-1])


def fetch_intraday_series(ticker: str) -> pd.Series:
    d = yf.download(
        ticker,
        period=INTRA_PERIOD,
        interval=INTRA_INTERVAL,
        auto_adjust=False,
        progress=False,
    )
    if d is None or d.empty:
        raise RuntimeError(f"[WARN] intraday empty for {ticker}")

    s = ensure_series_1d_close(d)

    # インデックスを JST に変換
    idx = pd.to_datetime(s.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert(BASE_TZ)

    # 当日（JST）だけ抽出
    last_day = idx[-1].date()
    mask_day = (idx.date == last_day)
    s = pd.Series(s.values, index=idx)[mask_day]

    if s.empty:
        raise RuntimeError(f"[WARN] intraday filtered empty for {ticker} (day)")

    # 当日の場中（09:00 - 15:30）に限定
    def in_session(ts: pd.Timestamp) -> bool:
        t = ts.timetz()
        # timetz()のtz情報はJST、比較は time オブジェクトでOK
        return (SESSION_START <= dtime(t.hour, t.minute) <= SESSION_END)

    s = s[[in_session(ts) for ts in s.index]]
    if s.empty:
        raise RuntimeError(f"[WARN] intraday filtered empty for {ticker} (session)")

    return s


# ---------- 指数構築 ----------
def build_equal_weight_index(tickers: List[str]) -> pd.DataFrame:
    rows = []
    for i, t in enumerate(tickers, 1):
        try:
            print(f"[INFO] ({i}/{len(tickers)}) Fetching {t} ...")
            prev = fetch_prev_close(t)
            intraday = fetch_intraday_series(t)
            pct = (intraday / prev - 1.0) * 100.0
            rows.append(pct.rename(t))
        except Exception as e:
            print(f"[WARN] skip {t}  # {e}")
        finally:
            time.sleep(FETCH_PAUSE_SEC)  # レート/キャッシュ誤命中回避

    if not rows:
        raise RuntimeError("取得できた日中データが0でした。ティッカーを見直してください。")

    # 時系列方向で結合（5分足のズレは mean(skipna=True) で吸収）
    df = pd.concat(rows, axis=1).sort_index()
    df["R_BANK9"] = df.mean(axis=1, skipna=True)
    return df


# ---------- 可視化 ----------
def pick_line_color(series: pd.Series) -> str:
    return "#00e5d7" if len(series) and float(series.iloc[-1]) >= 0 else "#ff4d4d"


def plot_index(df: pd.DataFrame) -> None:
    _ensure_outdir()
    series = df["R_BANK9"]
    c = pick_line_color(series)

    # 体裁統一（既存の黒基調）
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
    title = f"R-BANK9 Intraday Snapshot ({jst_now().strftime('%Y/%m/%d %H:%M')})"
    ax.set_title(title, color="white", fontsize=22, pad=12)
    ax.set_xlabel("Time", color="white")
    ax.set_ylabel("Change vs Prev Close (%)", color="white")
    ax.legend(facecolor="black", edgecolor="#444444", labelcolor="white", loc="upper left")
    fig.tight_layout()
    plt.savefig(IMG_PATH, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_csv(df: pd.DataFrame) -> None:
    _ensure_outdir()
    df.to_csv(CSV_PATH, encoding="utf-8", index_label="datetime_jst")


def save_post_text(df: pd.DataFrame, tickers: List[str]) -> None:
    _ensure_outdir()
    last = float(df["R_BANK9"].iloc[-1])
    sign = "🔺" if last >= 0 else "🔻"
    with open(POST_PATH, "w", encoding="utf-8") as f:
        f.write(
            f"{sign} R-BANK9 日中スナップショット（{jst_now().strftime('%Y/%m/%d %H:%M')}）\n"
            f"{last:+.2f}%（前日終値比）\n"
            f"※ 構成9銘柄の等ウェイト\n"
            f"#地方銀行 #R_BANK9 #日本株\n"
        )


def save_stats(df: pd.DataFrame) -> None:
    _ensure_outdir()
    payload = {
        "index_key": "R_BANK9",
        "label": "R-BANK9",
        "pct_intraday": float(df["R_BANK9"].iloc[-1]) / 100.0,  # ratioで保持（サイト側と整合）
        "basis": "prev_close",
        "session": {"start": "09:00", "end": "15:30", "anchor": "09:00"},
        "updated_at": jst_now().isoformat(),
    }
    import json
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ---------- メイン ----------
def main():
    tickers = load_tickers(TICKER_FILE)
    print("[INFO] Building R_BANK9 intraday index ...")
    df = build_equal_weight_index(tickers)
    # 出力
    plot_index(df)
    save_csv(df)
    save_post_text(df, tickers)
    save_stats(df)
    print("[INFO] done.")


if __name__ == "__main__":
    main()
