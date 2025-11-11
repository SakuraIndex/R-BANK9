# -*- coding: utf-8 -*-
"""
R-BANK9 intraday index snapshot (robust)
- 9 銘柄を等ウェイト
- yfinance 5m（7d window）から当日(JST)のみ抽出
- 異常値を ±20% でクリップ
- 共通グリッドに揃えて ffill
- 出力:
    ・docs/outputs/rbank9_intraday.csv   ... ts,pct（サイト側が読む形式）
    ・docs/outputs/rbank9_intraday.png   ... 参考PNG（任意）
    ・docs/outputs/rbank9_post_intraday.txt
    ・docs/outputs/rbank9_stats.json     ... 補助（任意）
"""

from __future__ import annotations
import os, json
from typing import List, Dict, Optional
from datetime import datetime, timezone, timedelta

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ---------- 設定 ----------
BASE_TZ = timezone(timedelta(hours=9))  # JST
OUT_DIR = "docs/outputs"
TICKER_FILE = "docs/tickers_rbank9.txt"   # 5830.T などが1行1ティッカー

TS_CSV = os.path.join(OUT_DIR, "rbank9_intraday.csv")      # ts,pct（サイト用）
IMG_PATH = os.path.join(OUT_DIR, "rbank9_intraday.png")    # 参考画像
POST_PATH = os.path.join(OUT_DIR, "rbank9_post_intraday.txt")
STATS_JSON = os.path.join(OUT_DIR, "rbank9_stats.json")

INTRA_PERIOD = "7d"
INTRA_INTERVAL = "5m"

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
    if not xs:
        raise RuntimeError("ティッカーリストが空です")
    return xs

def _ensure_series_1d_close(df: pd.DataFrame) -> pd.Series:
    if "Close" not in df.columns:
        raise ValueError("Close column not found")
    close = df["Close"]
    if isinstance(close, pd.Series):
        return pd.to_numeric(close, errors="coerce").dropna()
    d2 = close.apply(pd.to_numeric, errors="coerce")
    d2 = d2.loc[:, d2.notna().any(axis=0)]
    if d2.shape[1] == 0:
        raise ValueError("no numeric close column")
    if d2.shape[1] == 1:
        ser = d2.iloc[:, 0]
    else:
        ser = d2[d2.count(axis=0).idxmax()]
    return ser.dropna().astype(float)

def _download_prev_close(ticker: str) -> Optional[float]:
    d = yf.download(ticker, period="10d", interval="1d",
                    auto_adjust=False, progress=False)
    if d.empty:
        return None
    s = _ensure_series_1d_close(d)
    if s.empty:
        return None
    return float(s.iloc[-2] if len(s) >= 2 else s.iloc[-1])

def _download_intraday_today(ticker: str) -> pd.Series:
    d = yf.download(ticker, period=INTRA_PERIOD, interval=INTRA_INTERVAL,
                    auto_adjust=False, progress=False)
    if d.empty:
        return pd.Series(dtype=float)
    s = _ensure_series_1d_close(d)
    idx = pd.to_datetime(s.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert(BASE_TZ)
    s = pd.Series(s.values, index=idx)
    if s.empty:
        return pd.Series(dtype=float)
    # 直近セッション日に限定（場外でも最新日の当日分が取れる）
    last_day = idx[-1].date()
    s = s[s.index.date == last_day]
    return s

# ---------- 指数構築 ----------
def build_equal_weight_index(tickers: List[str]) -> pd.Series:
    indiv: Dict[str, pd.Series] = {}
    last_times: List[pd.Timestamp] = []

    for t in tickers:
        try:
            prev = _download_prev_close(t)
            intraday = _download_intraday_today(t)
            if prev is None or intraday.empty:
                print(f"[WARN] skip {t} (prev or intraday empty)")
                continue
            pct = (intraday / prev - 1.0) * 100.0
            pct = pct.clip(lower=PCT_CLIP_LOW, upper=PCT_CLIP_HIGH)
            indiv[t] = pct.rename(t)
            last_times.append(pct.index.max())
        except Exception as e:
            print(f"[WARN] skip {t}  # {e}")

    if not indiv:
        return pd.Series(dtype=float)

    # 共通終了時刻
    common_end = min(last_times)
    start_time = min(s.index.min() for s in indiv.values())
    grid = pd.date_range(start=start_time, end=common_end,
                         freq=INTRA_INTERVAL, tz=BASE_TZ)

    aligned = []
    for t, ser in indiv.items():
        s2 = ser.reindex(grid).ffill()
        aligned.append(s2.rename(t))
    df = pd.concat(aligned, axis=1)
    series = df.mean(axis=1, skipna=True).rename("R_BANK9")
    return series.dropna()

# ---------- 出力 ----------
def save_ts_pct(series: pd.Series) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    if series is None or series.empty:
        # 空でもヘッダだけ書く（サイト側が no data を描ける）
        with open(TS_CSV, "w", encoding="utf-8") as f:
            f.write("ts,pct\n")
        return
    d = pd.DataFrame({
        "ts": series.index.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pct": series.values
    })
    d.to_csv(TS_CSV, index=False, encoding="utf-8")

def plot_reference_png(series: pd.Series) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.close("all")
    fig = plt.figure(figsize=(16, 9), dpi=160)
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    for sp in ax.spines.values():
        sp.set_color("#444")
    if series is not None and not series.empty:
        c = "#00e5d7" if float(series.iloc[-1]) >= 0 else "#ff4d4d"
        ax.plot(series.index, series.values, color=c, linewidth=2.2, label="R-BANK9")
    ax.axhline(0, color="#666", linewidth=1.0)
    ax.tick_params(colors="white")
    ax.set_title(f"R-BANK9 Intraday Snapshot ({jst_now().strftime('%Y/%m/%d %H:%M')})",
                 color="white", fontsize=20, pad=10)
    ax.set_xlabel("Time", color="white")
    ax.set_ylabel("Change vs Prev Close (%)", color="white")
    ax.legend(facecolor="black", edgecolor="#444", labelcolor="white", loc="upper left")
    fig.tight_layout()
    plt.savefig(IMG_PATH, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

def save_post(series: pd.Series) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    if series is None or series.empty:
        txt = "▲ R-BANK9 日中スナップショット（no data）\n+0.00%（前日終値比）\n#地方銀行 #R_BANK9 #日本株\n"
    else:
        v = float(series.iloc[-1])
        sign = "🔺" if v >= 0 else "🔻"
        txt = (f"{sign} R-BANK9 日中スナップショット（{jst_now().strftime('%Y/%m/%d %H:%M')}）\n"
               f"{v:+.2f}%（前日終値比）\n"
               f"※ 構成9銘柄の等ウェイト\n"
               f"#地方銀行 #R_BANK9 #日本株\n")
    with open(POST_PATH, "w", encoding="utf-8") as f:
        f.write(txt)

def save_stats(series: pd.Series) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    val = 0.0 if series is None or series.empty else float(series.iloc[-1])
    payload = {
        "index_key": "rbank9",
        "label": "R-BANK9",
        "pct_intraday": val,
        "updated_at": jst_now().isoformat()
    }
    with open(STATS_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

# ---------- メイン ----------
def main():
    tickers = load_tickers(TICKER_FILE)
    print("[INFO] Building R_BANK9 intraday ...")
    series = build_equal_weight_index(tickers)

    # ts,pct（サイトが読む CSV）
    save_ts_pct(series)
    # 参考PNG / 投稿文 / stats
    plot_reference_png(series)
    save_post(series)
    save_stats(series)
    print("[INFO] done.")

if __name__ == "__main__":
    main()
