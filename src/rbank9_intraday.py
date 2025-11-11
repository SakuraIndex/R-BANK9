# -*- coding: utf-8 -*-
"""
R-BANK9 intraday index snapshot

- 9銘柄等ウェイトの前日終値比（%）を5分足で推定
- 安全弁（±20%でクリップ）、共通グリッドに整列して ffill
- 出力:
    docs/outputs/rbank9_intraday.csv        ... ts,pct の２列（JST, ISO8601, サイト用）
    docs/outputs/rbank9_intraday_full.csv   ... デバッグ用（各銘柄列＋R_BANK9）
    docs/outputs/rbank9_post_intraday.txt   ... 簡易テキスト
    docs/outputs/rbank9_intraday.png        ... 参考用（リポ内表示向け）
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

PNG_PATH = os.path.join(OUT_DIR, "rbank9_intraday.png")
CSV_SITE = os.path.join(OUT_DIR, "rbank9_intraday.csv")         # ← ts,pct（サイトが読む）
CSV_FULL = os.path.join(OUT_DIR, "rbank9_intraday_full.csv")    # ← 多列（デバッグ）

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
    """yfinance の Close を 1 次元 Series[float] に正規化"""
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
    idx = idx.tz_convert("Asia/Tokyo")
    last_day = idx[-1].date()
    s = pd.Series(s.values, index=idx)
    s = s[(s.index.date == last_day)]
    if s.empty:
        raise RuntimeError(f"[WARN] intraday filtered empty for {ticker}")
    return s

# ---------- 指数構築 ----------
def build_equal_weight_index(tickers: List[str]) -> pd.DataFrame:
    indiv_pct: Dict[str, pd.Series] = {}
    last_times: List[pd.Timestamp] = []

    for t in tickers:
        try:
            print(f"[INFO] Fetching {t} ...")
            prev = fetch_prev_close(t)
            intraday = fetch_intraday_series(t)
            pct = (intraday / prev - 1.0) * 100.0
            pct = pct.clip(lower=PCT_CLIP_LOW, upper=PCT_CLIP_HIGH)
            indiv_pct[t] = pct.rename(t)
            last_times.append(pct.index.max())
        except Exception as e:
            print(f"[WARN] skip {t}  # {e}")

    if not indiv_pct:
        raise RuntimeError("取得できた日中データが0でした。ティッカーを見直してください。")

    # 共通に揃う終了時刻（全銘柄の最終TSの最小値）
    common_end = min(last_times)
    start_time = min(s.index.min() for s in indiv_pct.values())

    grid = pd.date_range(start=start_time, end=common_end,
                         freq=INTRA_INTERVAL, tz="Asia/Tokyo")

    aligned = []
    for t, ser in indiv_pct.items():
        s2 = ser.reindex(grid).ffill()
        aligned.append(s2.rename(t))

    df = pd.concat(aligned, axis=1)
    df["R_BANK9"] = df.mean(axis=1, skipna=True)
    df = df.sort_index()
    return df

# ---------- 保存系 ----------
def save_csvs(df: pd.DataFrame) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) サイト用（ts,pct）
    s = df["R_BANK9"].dropna()
    site = pd.DataFrame({
        "ts": s.index.tz_convert("Asia/Tokyo").astype("datetime64[ns, Asia/Tokyo]"),
        "pct": s.values
    })
    # ISO8601(+09:00)で保存
    site["ts"] = site["ts"].dt.tz_convert("Asia/Tokyo").dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    # +0900 -> +09:00 へ整形
    site["ts"] = site["ts"].str.replace(r"(\+|\-)(\d{2})(\d{2})$", r"\1\2:\3", regex=True)
    site.to_csv(CSV_SITE, index=False, header=False, encoding="utf-8")

    # 2) デバッグ用フル（各銘柄＋R_BANK9）
    df.to_csv(CSV_FULL, encoding="utf-8")

def save_post_text(df: pd.DataFrame) -> None:
    last = float(df["R_BANK9"].iloc[-1])
    sign = "🔺" if last >= 0 else "🔻"
    with open(POST_PATH, "w", encoding="utf-8") as f:
        f.write(
            f"{sign} R-BANK9 日中スナップショット（{jst_now().strftime('%Y/%m/%d %H:%M')} JST）\n"
            f"{last:+.2f}%（前日終値比）\n"
            f"※ 構成9銘柄の等ウェイト\n"
            f"#地方銀行 #R_BANK9 #日本株\n"
        )

def plot_index(df: pd.DataFrame) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    series = df["R_BANK9"]
    c = "#00e5d7" if float(series.iloc[-1]) >= 0 else "#ff4d4d"

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
        f"R-BANK9 Intraday Snapshot ({jst_now().strftime('%Y/%m/%d %H:%M')} JST)",
        color="white", fontsize=22, pad=12
    )
    ax.set_xlabel("Time", color="white")
    ax.set_ylabel("Change vs Prev Close (%)", color="white")
    ax.legend(facecolor="black", edgecolor="#444444", labelcolor="white", loc="upper left")
    fig.tight_layout()
    plt.savefig(PNG_PATH, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


# ---------- メイン ----------
def main():
    tickers = load_tickers(TICKER_FILE)
    print("[INFO] Building R_BANK9 intraday index ...")
    df = build_equal_weight_index(tickers)
    save_csvs(df)          # ← サイト用 ts,pct を書く
    plot_index(df)         # 参考PNG
    save_post_text(df)     # 参考TXT
    print("[INFO] done.")

if __name__ == "__main__":
    main()
