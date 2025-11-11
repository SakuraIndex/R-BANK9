# -*- coding: utf-8 -*-
"""
R-BANK9 intraday index snapshot
- 等ウェイト / 前日終値比（%）で当日チャートを作成
- 取得失敗や市場時間外でも、必ず ts,pct を出力（フォールバック実装）

出力:
  docs/outputs/rbank9_intraday.csv   … ts,pct
  docs/outputs/rbank9_intraday.png   … スナップショット
  docs/outputs/rbank9_post_intraday.txt
"""

from __future__ import annotations
import os
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone, timedelta

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ---------- 設定 ----------
BASE_TZ = timezone(timedelta(hours=9))  # JST
OUT_DIR = "docs/outputs"
TICKER_FILE = "docs/tickers_rbank9.txt"   # 5830.T などを1行1ティッカー

PNG_PATH = os.path.join(OUT_DIR, "rbank9_intraday.png")
CSV_PATH = os.path.join(OUT_DIR, "rbank9_intraday.csv")
POST_PATH = os.path.join(OUT_DIR, "rbank9_post_intraday.txt")

# まずは5分足、それで空なら15分足で再トライ
PRIMARY_PERIOD = "5d"
PRIMARY_INTERVAL = "5m"
SECONDARY_PERIOD = "30d"
SECONDARY_INTERVAL = "15m"

# 異常値クリップ（安全弁）
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

def _to_series_numeric(close_like: pd.DataFrame | pd.Series, idx) -> pd.Series:
    """yfinanceの戻りを数値Seriesに正規化"""
    if isinstance(close_like, pd.Series):
        ser = pd.to_numeric(close_like, errors="coerce").dropna()
        ser.index = idx
        return ser

    df = close_like.apply(pd.to_numeric, errors="coerce")
    df = df.loc[:, df.notna().any(axis=0)]
    if df.shape[1] == 0:
        return pd.Series([], dtype=float)

    if df.shape[1] == 1:
        ser = df.iloc[:, 0]
    else:
        best = df.count(axis=0).idxmax()
        ser = df[best]
    ser = pd.to_numeric(ser, errors="coerce").dropna()
    ser.index = idx[: len(ser)]
    return ser

def ensure_close_series(d: pd.DataFrame) -> pd.Series:
    if "Close" not in d.columns:
        return pd.Series([], dtype=float)
    return _to_series_numeric(d["Close"], d.index)

def fetch_prev_close(ticker: str) -> Optional[float]:
    d = yf.download(ticker, period="10d", interval="1d",
                    auto_adjust=False, progress=False)
    if d is None or d.empty:
        return None
    s = ensure_close_series(d)
    if s.empty:
        return None
    if len(s) >= 2:
        return float(s.iloc[-2])
    return float(s.iloc[-1])

def _fetch_intraday_once(ticker: str, period: str, interval: str) -> pd.Series:
    d = yf.download(ticker, period=period, interval=interval,
                    auto_adjust=False, progress=False)
    if d is None or d.empty:
        return pd.Series([], dtype=float)

    s = ensure_close_series(d)
    if s.empty:
        return s

    idx = pd.to_datetime(s.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert(BASE_TZ)
    s = pd.Series(s.values, index=idx)
    return s

def fetch_intraday_series_jst(ticker: str) -> pd.Series:
    """5m→だめなら15m。直近営業日のみ抽出して返す"""
    for period, interval in [(PRIMARY_PERIOD, PRIMARY_INTERVAL),
                             (SECONDARY_PERIOD, SECONDARY_INTERVAL)]:
        s = _fetch_intraday_once(ticker, period, interval)
        if s.empty:
            continue
        # “直近の営業日”を抽出（最終時刻の date）
        last_day = s.index[-1].date()
        ss = s[s.index.date == last_day]
        if not ss.empty:
            return ss
    return pd.Series([], dtype=float)

# ---------- 指数構築 ----------
def build_equal_weight_index(tickers: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """当日(直近営業日)の等ウェイト％シリーズを組む。返り値: df, used_tickers"""
    indiv_pct: Dict[str, pd.Series] = {}
    used: List[str] = []
    last_times: List[pd.Timestamp] = []

    for t in tickers:
        try:
            prev = fetch_prev_close(t)
            if prev is None or prev == 0:
                print(f"[WARN] prev close empty for {t}")
                continue

            intraday = fetch_intraday_series_jst(t)
            if intraday.empty:
                print(f"[WARN] intraday empty for {t}")
                continue

            pct = (intraday / float(prev) - 1.0) * 100.0
            pct = pct.clip(lower=PCT_CLIP_LOW, upper=PCT_CLIP_HIGH)
            indiv_pct[t] = pct.rename(t)
            used.append(t)
            last_times.append(pct.index.max())
            print(f"[INFO] ok: {t}, points={len(pct)}")
        except Exception as e:
            print(f"[WARN] skip {t}  # {e}")

    if not indiv_pct:
        return pd.DataFrame(), []

    # 共通終了時刻に合わせる
    common_end = min(last_times)
    start_time = min(s.index.min() for s in indiv_pct.values())
    grid = pd.date_range(start=start_time, end=common_end,
                         freq=PRIMARY_INTERVAL, tz=BASE_TZ)

    aligned = []
    for t, ser in indiv_pct.items():
        s2 = ser.reindex(grid).ffill()
        aligned.append(s2.rename(t))

    df = pd.concat(aligned, axis=1)
    df["R_BANK9"] = df.mean(axis=1, skipna=True)
    df = df.dropna(subset=["R_BANK9"])
    return df, used

def fallback_from_daily(tickers: List[str]) -> pd.DataFrame:
    """最悪時：日足から当日の等ウェイト％を一つの点として作り、5分グリッドに水平線を引く"""
    vals = []
    for t in tickers:
        try:
            d = yf.download(t, period="10d", interval="1d",
                            auto_adjust=False, progress=False)
            if d is None or d.empty:
                continue
            close = ensure_close_series(d)
            if len(close) < 2:
                continue
            pct = (float(close.iloc[-1]) / float(close.iloc[-2]) - 1.0) * 100.0
            vals.append(max(min(pct, PCT_CLIP_HIGH), PCT_CLIP_LOW))
        except Exception:
            continue

    if not vals:
        return pd.DataFrame()

    eq = sum(vals) / len(vals)
    # 今日の 09:00–15:30 を 5分グリッドにして水平線
    today = jst_now().date()
    idx = pd.date_range(
        start=pd.Timestamp(today, tz=BASE_TZ) + timedelta(hours=9),
        end=pd.Timestamp(today, tz=BASE_TZ) + timedelta(hours=15, minutes=30),
        freq=PRIMARY_INTERVAL,
        tz=BASE_TZ,
    )
    series = pd.Series([eq] * len(idx), index=idx, name="R_BANK9")
    return pd.DataFrame({"R_BANK9": series})

# ---------- 出力 ----------
def write_ts_pct_csv(series: pd.Series, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ts,pct\n")
        for ts, v in series.items():
            # ISO文字列（タイムゾーン付き）で
            f.write(f"{pd.Timestamp(ts).isoformat()},{float(v):.6f}\n")

def plot_png(series: pd.Series) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.close("all")
    fig = plt.figure(figsize=(16, 9), dpi=160)
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor("#0b1220")
    ax.set_facecolor("#0b1220")
    for sp in ax.spines.values():
        sp.set_visible(False)
    color = "#00e5d7" if len(series) and float(series.iloc[-1]) >= 0 else "#ff4d4d"
    if not series.empty:
        ax.plot(series.index, series.values, color=color, linewidth=2.2, label="R-BANK9")
    ax.axhline(0, color="#475569", linewidth=1.0)
    ax.tick_params(colors="#d1d5db")
    ax.set_title(f"R-BANK9 Intraday Snapshot ({jst_now().strftime('%Y/%m/%d %H:%M JST')})",
                 color="#d1d5db", fontsize=18, pad=10)
    ax.set_xlabel("Time", color="#d1d5db")
    ax.set_ylabel("Change vs Prev Close (%)", color="#d1d5db")
    fig.tight_layout()
    plt.savefig(PNG_PATH, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

def write_post(series: pd.Series, used_n: int) -> None:
    last = 0.0 if series.empty else float(series.iloc[-1])
    sign = "+" if last >= 0 else ""
    note = "" if not series.empty else "（no data / fallback）"
    txt = (
        f"▲ R-BANK9 日中スナップショット（{jst_now().strftime('%Y/%m/%d %H:%M JST')}）{note}\n"
        f"{sign}{last:.2f}%（前日終値比）\n"
        f"※ 構成{used_n if used_n>0 else 9}銘柄の等ウェイト\n"
        "#地方銀行 #R_BANK9 #日本株\n"
    )
    with open(POST_PATH, "w", encoding="utf-8") as f:
        f.write(txt)

# ---------- メイン ----------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tickers = load_tickers(TICKER_FILE)
    if not tickers:
        print("[ERROR] no tickers")
        # 空でもフォーマットを出しておく
        with open(CSV_PATH, "w", encoding="utf-8") as f:
            f.write("ts,pct\n")
        write_post(pd.Series(dtype=float), 0)
        return

    print("[INFO] building intraday …")
    df, used = build_equal_weight_index(tickers)

    if df.empty:
        print("[WARN] intraday build failed → fallback from daily")
        df = fallback_from_daily(tickers)
        used = tickers if not df.empty else []

    series = df["R_BANK9"] if ("R_BANK9" in df.columns) else pd.Series(dtype=float)

    # CSV(ts,pct) は必ず出力
    if series.empty:
        with open(CSV_PATH, "w", encoding="utf-8") as f:
            f.write("ts,pct\n")
    else:
        write_ts_pct_csv(series, CSV_PATH)

    # PNG とポスト
    plot_png(series)
    write_post(series, len(used))
    print(f"[INFO] done. points={len(series)} used={len(used)}")

if __name__ == "__main__":
    main()
