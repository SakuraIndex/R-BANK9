# -*- coding: utf-8 -*-
"""
R-BANK9 intraday index snapshot
等ウェイト / 前日終値比（%）で1日チャートを描画（黒背景・SNS向け）
- タイムスタンプのズレで終盤だけ一部銘柄しか揃わない問題を修正
- 固定アンカー（09:00〜15:25 JST, 5分足）へ整列し、同バー内は ffill
- 十分な銘柄数が揃わないバーは前値維持でスパイク抑制
"""

from __future__ import annotations

import os
from math import ceil
from typing import Dict, List
from datetime import datetime, timezone, timedelta

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ---------- 設定 ----------
JST = timezone(timedelta(hours=9))
OUT_DIR = "docs/outputs"
TICKER_FILE = "docs/tickers_rbank9.txt"  # 5830.T などを1行1ティッカー

IMG_PATH = os.path.join(OUT_DIR, "rbank9_intraday.png")
CSV_PATH = os.path.join(OUT_DIR, "rbank9_intraday.csv")
POST_PATH = os.path.join(OUT_DIR, "rbank9_post_intraday.txt")
STATS_PATH = os.path.join(OUT_DIR, "rbank9_stats.json")

# yfinance（JP）は 1m が不安定なことがあるので 5m を使用
INTRA_PERIOD = "7d"
INTRA_INTERVAL = "5m"

# 十分なカバレッジとみなす最小銘柄数（全体の 60%）
MIN_COVERAGE_RATIO = 0.60


# ---------- ユーティリティ ----------
def jst_now() -> datetime:
    return datetime.now(JST)


def load_tickers(path: str) -> List[str]:
    xs: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            xs.append(s)
    if not xs:
        raise RuntimeError("ティッカーリストが空です。")
    return xs


def _ensure_outdir() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)


def _to_close_series_1d(frame_or_series: pd.DataFrame | pd.Series, index) -> pd.Series:
    """
    yfinance の Close が (N,), (N,1), (N,k) など何で来ても 1 次元 Series[float] に正規化。
    """
    if isinstance(frame_or_series, pd.Series):
        s = pd.to_numeric(frame_or_series, errors="coerce").astype(float).dropna()
        s.index = index
        return s

    df = frame_or_series.apply(pd.to_numeric, errors="coerce")
    df = df.loc[:, df.notna().any(axis=0)]
    if df.shape[1] == 0:
        raise ValueError("Close が取得できませんでした。")
    if df.shape[1] == 1:
        s = df.iloc[:, 0]
    else:
        best = df.count(axis=0).idxmax()
        s = df[best]
    s = s.astype(float).dropna()
    s.index = index
    return s


def _ensure_close_series(df: pd.DataFrame) -> pd.Series:
    if "Close" not in df.columns:
        raise ValueError("Close column not found")
    return _to_close_series_1d(df["Close"], df.index)


def fetch_prev_close(ticker: str) -> float:
    d = yf.download(ticker, period="10d", interval="1d", auto_adjust=False, progress=False)
    if d.empty:
        raise RuntimeError(f"[WARN] prev close empty for {ticker}")
    s = _ensure_close_series(d)
    # 前日終値（直近1本前、なければ直近）
    return float(s.iloc[-2] if len(s) >= 2 else s.iloc[-1])


def fetch_intraday_close_today(ticker: str) -> pd.Series:
    """
    当日(JST)の 5 分足 Close を JST タイムゾーンの Series で返す
    """
    d = yf.download(
        ticker, period=INTRA_PERIOD, interval=INTRA_INTERVAL, auto_adjust=False, progress=False
    )
    if d.empty:
        raise RuntimeError(f"[WARN] intraday empty for {ticker}")
    s = _ensure_close_series(d)

    # Index -> JST
    idx = pd.to_datetime(s.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert(JST)
    s.index = idx

    # 当日分に限定
    today = jst_now().date()
    s = s[idx.date == today]
    if s.empty:
        raise RuntimeError(f"[WARN] intraday filtered empty for {ticker}")
    return s


def build_anchor_index() -> pd.DatetimeIndex:
    """
    当日(JST) 09:00〜15:25 の 5分足固定アンカー。
    """
    d = jst_now().date()
    start = datetime(d.year, d.month, d.day, 9, 0, tzinfo=JST)
    end = datetime(d.year, d.month, d.day, 15, 25, tzinfo=JST)
    return pd.date_range(start=start, end=end, freq="5T", tz=JST)


# ---------- 指数構築 ----------
def build_equal_weight_index(tickers: List[str]) -> pd.DataFrame:
    """
    1) 各銘柄の当日 5分足 Close を取得
    2) 5分でfloorして同一バーに整列 → 固定アンカーへ reindex + ffill（同日内）
    3) 前日終値比(%)を算出
    4) カバレッジ不十分なバーは前値で維持（スパイク抑制）
    """
    anchor = build_anchor_index()
    series_map: Dict[str, pd.Series] = {}
    prev_map: Dict[str, float] = {}

    for t in tickers:
        try:
            prev = fetch_prev_close(t)
            s = fetch_intraday_close_today(t)

            # 5分へ丸め（floor）して同一バーの最後を採用
            s = (
                s.to_frame("Close")
                .assign(bin=lambda df: df.index.floor("5T"))
                .groupby("bin")["Close"]
                .last()
            )

            # 固定アンカーへ合わせ、同日内での前回値を使用（method='ffill'）
            s = s.reindex(anchor).ffill()

            series_map[t] = s
            prev_map[t] = prev
        except Exception as e:
            print(f"[WARN] skip {t}  # {e}")

    if not series_map:
        raise RuntimeError("取得できた日中データが0でした。ティッカーを見直してください。")

    # ピボット（列=銘柄, 行=アンカー時刻）
    close_df = pd.DataFrame(series_map).reindex(anchor)

    # 各銘柄の前日終値で % に変換
    pct_df = pd.DataFrame(
        {t: (close_df[t] / prev_map[t] - 1.0) * 100.0 for t in close_df.columns},
        index=close_df.index,
    )

    # カバレッジ判定：値が入っている列の数
    count = pct_df.notna().sum(axis=1)
    min_cov = ceil(len(tickers) * MIN_COVERAGE_RATIO)

    # 等ウェイト平均（揃わないバーは NaN→前値維持でスパイク抑制）
    eq_mean = pct_df.mean(axis=1, skipna=True)
    eq_mean = eq_mean.where(count >= min_cov).ffill()

    out = pct_df.copy()
    out["R_BANK9"] = eq_mean
    return out


# ---------- 可視化 ----------
def pick_line_color(series: pd.Series) -> str:
    return "#00e5d7" if len(series) and float(series.iloc[-1]) >= 0 else "#ff4d4d"


def plot_index(df: pd.DataFrame) -> None:
    _ensure_outdir()
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
    title_date = jst_now().strftime("%Y/%m/%d %H:%M")
    ax.set_title(f"R-BANK9 Intraday Snapshot ({title_date})", color="white", fontsize=22, pad=12)
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
    last = float(df["R_BANK9"].iloc[-1])
    sign = "🔺" if last >= 0 else "🔻"
    with open(POST_PATH, "w", encoding="utf-8") as f:
        f.write(
            f"{sign} R-BANK9 日中スナップショット（{jst_now().strftime('%Y/%m/%d %H:%M')}）\n"
            f"{last:+.2f}%（前日終値比）\n"
            f"※ 構成{len(tickers)}銘柄の等ウェイト\n"
            f"#地方銀行 #R_BANK9 #日本株\n"
        )


def save_stats(df: pd.DataFrame) -> None:
    _ensure_outdir()
    last = float(df["R_BANK9"].iloc[-1])
    stats = {
        "index_key": "R_BANK9",
        "label": "R-BANK9",
        "pct_intraday": last / 100.0,  # 小数（0.076など）
        "basis": "prev_close",
        "session": {"start": "09:00", "end": "15:30", "anchor": "09:00"},
        "updated_at": jst_now().isoformat(),
    }
    pd.Series(stats).to_json(STATS_PATH, force_ascii=False)


# ---------- メイン ----------
def main():
    tickers = load_tickers(TICKER_FILE)
    print("[INFO] Building R_BANK9 intraday index ...")
    df = build_equal_weight_index(tickers)
    save_csv(df)
    plot_index(df)
    save_post_text(df, tickers)
    save_stats(df)
    print("[INFO] done.")


if __name__ == "__main__":
    main()
