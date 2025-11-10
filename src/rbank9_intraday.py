# -*- coding: utf-8 -*-
"""
R-BANK9 intraday index snapshot
等ウェイト / 前日終値比（%）で1日チャートを描画（黒背景・SNS向け）
"""

import os
from typing import List
from datetime import datetime, timezone, timedelta, time as dtime

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ---------- 設定 ----------
JST = timezone(timedelta(hours=9))  # JST
OUT_DIR = "docs/outputs"
TICKER_FILE = os.path.join(OUT_DIR, "tickers_rbank9.txt")  # 1行1ティッカー

IMG_PATH = os.path.join(OUT_DIR, "rbank9_intraday.png")
CSV_PATH = os.path.join(OUT_DIR, "rbank9_intraday.csv")
POST_PATH = os.path.join(OUT_DIR, "rbank9_post_intraday.txt")
STATS_PATH = os.path.join(OUT_DIR, "rbank9_stats.json")

# JP は 1m が不安定なことがあるので 5m で安定運用
INTRA_PERIOD = "7d"
INTRA_INTERVAL = "5m"

# 取引時間（JST）
SESSION_START = dtime(9, 0)
SESSION_END = dtime(15, 30)

# 行を採用するために必要な最小の有効銘柄率（例: 0.7 で全体の70%以上にデータがある行だけ採用）
ROW_MIN_COVER_RATIO = 0.7


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
        raise RuntimeError(f"No tickers in {path}")
    return xs


def _to_series_1d(close_like: pd.DataFrame | pd.Series, index) -> pd.Series:
    """
    yfinance の Close が (N,), (N,1), (N,k) など何で来ても 1次元 Series[float] に正規化する。
    """
    if isinstance(close_like, pd.Series):
        ser = pd.to_numeric(close_like, errors="coerce").dropna()
        ser.index = index
        return ser

    df = close_like.apply(pd.to_numeric, errors="coerce")
    df = df.loc[:, df.notna().any(axis=0)]  # 全欠損列を落とす

    if df.shape[1] == 0:
        raise ValueError("no numeric close column")

    if df.shape[1] == 1:
        ser = df.iloc[:, 0]
    else:
        # 有効データ点数が最も多い列を採用
        best_col = df.count(axis=0).idxmax()
        ser = df[best_col]

    ser = ser.astype(float)
    ser.index = index
    ser = ser.dropna()
    return ser


def ensure_series_1d_close(df: pd.DataFrame) -> pd.Series:
    if "Close" not in df.columns:
        raise ValueError("Close column not found")
    return _to_series_1d(df["Close"], df.index)


def fetch_prev_close(ticker: str) -> float:
    d = yf.download(ticker, period="10d", interval="1d",
                    auto_adjust=False, progress=False)
    if d.empty:
        raise RuntimeError(f"[WARN] prev close empty for {ticker}")
    s = ensure_series_1d_close(d)
    # 前日終値（直近1本前を優先）
    if len(s) >= 2:
        return float(s.iloc[-2])
    return float(s.iloc[-1])


def fetch_intraday_series(ticker: str) -> pd.Series:
    d = yf.download(ticker, period=INTRA_PERIOD, interval=INTRA_INTERVAL,
                    auto_adjust=False, progress=False)
    if d.empty:
        raise RuntimeError(f"[WARN] intraday empty for {ticker}")
    s = ensure_series_1d_close(d)

    # UTC -> JST / 当日（JST）のみ
    idx = pd.to_datetime(s.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert(JST)
    s.index = idx

    last_day = idx[-1].date()
    s = s[(s.index.date == last_day)]
    if s.empty:
        raise RuntimeError(f"[WARN] intraday filtered empty for {ticker}")

    # 取引時間でフィルタ
    s = s[(s.index.time >= SESSION_START) & (s.index.time <= SESSION_END)]
    return s


# ---------- 指数構築 ----------
def build_equal_weight_index(tickers: List[str]) -> pd.DataFrame:
    rows = []
    prev_map: dict[str, float] = {}
    for t in tickers:
        try:
            prev = fetch_prev_close(t)
            intraday = fetch_intraday_series(t)
            pct = (intraday / prev - 1.0) * 100.0  # % に変換
            rows.append(pct.rename(t))
            prev_map[t] = prev
        except Exception as e:
            print(f"[WARN] skip {t}  # {e}")

    if not rows:
        raise RuntimeError("取得できた日中データが0でした。ティッカーを見直してください。")

    # 列方向に結合・時刻ソート
    df = pd.concat(rows, axis=1).sort_index()

    # 少数銘柄だけが更新された時刻（例: 場後ティック）を除外
    min_count = max(1, int(len(tickers) * ROW_MIN_COVER_RATIO))
    cnt = df.count(axis=1)
    df = df.loc[cnt >= min_count]

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
    ax.fill_between(series.index, 0, series.values, where=series.values >= 0,
                    step="pre", alpha=0.15, color=c)
    ax.axhline(0, color="#666666", linewidth=1.0)
    ax.tick_params(colors="white")
    ax.set_title(f"R-BANK9 Intraday Snapshot ({jst_now().strftime('%Y/%m/%d %H:%M JST')})",
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


def save_post_text(df: pd.DataFrame) -> None:
    last = float(df["R_BANK9"].iloc[-1]) if len(df) else 0.0
    sign = "🔺" if last >= 0 else "🔻"
    with open(POST_PATH, "w", encoding="utf-8") as f:
        f.write(
            f"{sign} R-BANK9 日中スナップショット（{jst_now().strftime('%Y/%m/%d %H:%M JST')}）\n"
            f"{last:+.2f}%（前日終値比）\n"
            f"※ 構成9銘柄の等ウェイト\n"
            f"#地方銀行 #R_BANK9 #日本株\n"
        )


def save_stats(df: pd.DataFrame) -> None:
    """ダッシュボード用の軽量JSON。pct_intraday は % 値で保存。"""
    last = float(df["R_BANK9"].iloc[-1]) if len(df) else 0.0
    payload = {
        "index_key": "R_BANK9",
        "label": "R-BANK9",
        "pct_intraday": round(last, 6),   # ← 例: 2.345678 (%)
        "basis": "prev_close",
        "session": {
            "start": SESSION_START.strftime("%H:%M"),
            "end": SESSION_END.strftime("%H:%M"),
            "anchor": SESSION_START.strftime("%H:%M"),
        },
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
    plot_index(df)
    save_csv(df)
    save_post_text(df)
    save_stats(df)
    print("[INFO] done.")

if __name__ == "__main__":
    main()
