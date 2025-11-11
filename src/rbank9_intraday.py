# -*- coding: utf-8 -*-
"""
R-BANK9 intraday generator (CSV: ts,pct)

- 構成銘柄の当日(=JST) 5分足の終値を等ウェイトで集計
- 各銘柄は前日終値を基準に pct を計算
- 5分グリッド(JST)に reindex+ffill して整列 → 共通グリッドで平均
- ±20% でクリップして瞬間異常値を抑制
- 出力: docs/outputs/rbank9_intraday.csv  （ヘッダ: ts,pct）
       * ts は JST の ISO8601（+09:00）で書き出し
"""

from __future__ import annotations

import os
from typing import List, Dict
from datetime import datetime, timezone, timedelta, time

import pandas as pd
import yfinance as yf

# ===== 基本設定 =====
JST = timezone(timedelta(hours=9))
OUT_DIR = "docs/outputs"
TICKER_FILE = "docs/tickers_rbank9.txt"   # 例: 5830.T を1行1件

CSV_PATH = os.path.join(OUT_DIR, "rbank9_intraday.csv")

# 取得パラメータ（JPは1mが不安定なことが多いので5m）
YF_PERIOD   = "7d"
YF_INTERVAL = "5m"

SESSION_START = time(9, 0)   # 09:00 JST
SESSION_END   = time(15, 30) # 15:30 JST

PCT_CLIP_LOW  = -20.0
PCT_CLIP_HIGH =  20.0


# ===== ユーティリティ =====
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
        raise RuntimeError("ティッカー一覧が空です。docs/tickers_rbank9.txt を確認してください。")
    return xs

def ensure_close_series_1d(df: pd.DataFrame) -> pd.Series:
    if "Close" not in df.columns:
        # yfinance の api 仕様により MultiIndex になることがあるため保険
        if isinstance(df.columns, pd.MultiIndex):
            # 一番上位が "Close" の列を探す
            for col in df.columns:
                if str(col[0]).lower() == "close":
                    s = pd.to_numeric(df[col], errors="coerce").dropna()
                    s.name = "Close"
                    return s
        raise ValueError("Close column not found")
    s = pd.to_numeric(df["Close"], errors="coerce").dropna()
    s.name = "Close"
    return s

def fetch_prev_close(ticker: str) -> float:
    d = yf.download(ticker, period="10d", interval="1d", auto_adjust=False, progress=False)
    if d.empty:
        raise RuntimeError(f"prev close empty: {ticker}")
    s = ensure_close_series_1d(d)
    if len(s) >= 2:
        return float(s.iloc[-2])
    return float(s.iloc[-1])

def fetch_intraday_close_today(ticker: str) -> pd.Series:
    d = yf.download(ticker, period=YF_PERIOD, interval=YF_INTERVAL, auto_adjust=False, progress=False)
    if d.empty:
        raise RuntimeError(f"intraday empty: {ticker}")
    s = ensure_close_series_1d(d)

    # index → JST
    idx = pd.to_datetime(s.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert(JST)
    s = pd.Series(s.values, index=idx)

    # 当日（JST）だけ
    today = jst_now().date()
    s = s[s.index.date == today]
    if s.empty:
        raise RuntimeError(f"intraday today empty: {ticker}")
    return s

def session_grid_for_today(freq: str = "5min") -> pd.DatetimeIndex:
    today = jst_now().date()
    start = datetime.combine(today, SESSION_START, tzinfo=JST)
    end   = datetime.combine(today, SESSION_END,   tzinfo=JST)
    return pd.date_range(start=start, end=end, freq=freq, tz=JST)

# ===== 集計 =====
def build_equal_weight_pct(tickers: List[str]) -> pd.Series:
    grid = session_grid_for_today("5min")
    aligned: List[pd.Series] = []
    for t in tickers:
        try:
            prev = fetch_prev_close(t)
            it   = fetch_intraday_close_today(t)
            pct  = (it / prev - 1.0) * 100.0
            pct  = pct.clip(lower=PCT_CLIP_LOW, upper=PCT_CLIP_HIGH)
            # セッション共通グリッドに合わせて ffill
            s2   = pct.reindex(grid).ffill()
            aligned.append(s2.rename(t))
            print(f"[ok] {t} points={len(s2.dropna())}")
        except Exception as e:
            print(f"[skip] {t}: {e}")

    if not aligned:
        raise RuntimeError("当日データを取得できた銘柄がありません。")

    df = pd.concat(aligned, axis=1)
    # 等ウェイト平均（銘柄ごとの欠損は skipna で除外）
    idx_pct = df.mean(axis=1, skipna=True)
    # ドロップ/重複整理（念のため）
    idx_pct = idx_pct.dropna()
    idx_pct = idx_pct[~idx_pct.index.duplicated(keep="last")]
    return idx_pct

# ===== 出力 =====
def write_ts_pct_csv(series_pct: pd.Series, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # 文字列化（JSTのISO8601）
    lines = ["ts,pct"]
    for ts, v in series_pct.items():
        # tz-aware を保証
        if ts.tzinfo is None:
            ts = ts.tz_localize(JST)
        lines.append(f"{ts.isoformat()},{float(v):.6f}")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    print(f"[write] {path} ({len(series_pct)} rows)")

# OPTIONAL: PNG を供給側でも出したい場合は有効化（サイト側で描くなら不要）
def plot_csv(series_pct: pd.Series, out_png: str) -> None:
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
        fig, ax = plt.subplots(figsize=(10, 4), dpi=140)
        fig.patch.set_facecolor("#0b1220")
        ax.set_facecolor("#0b1220")
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.grid(True, alpha=0.35, color="#334155")
        ax.plot(series_pct.index, series_pct.values, linewidth=1.8, color="#f87171")
        ax.set_title("R-BANK9 Intraday (JST)", color="#d1d5db")
        ax.set_xlabel("Time", color="#d1d5db")
        ax.set_ylabel("Change vs Prev Close (%)", color="#d1d5db")
        ax.tick_params(colors="#d1d5db")
        fig.tight_layout()
        fig.savefig(out_png, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        print(f"[write] {out_png}")
    except Exception as e:
        print(f"[warn] plot skipped: {e}")

# ===== メイン =====
def main():
    tickers = load_tickers(TICKER_FILE)
    series = build_equal_weight_pct(tickers)
    # セッション外の時間帯は空になり得る → CSVは必ず上書き（空ならヘッダのみ）
    if series.empty:
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(CSV_PATH, "w", encoding="utf-8", newline="\n") as f:
            f.write("ts,pct\n")
        print("[info] no points for today (session closed?) -> wrote header only")
        return

    write_ts_pct_csv(series, CSV_PATH)

    # 供給側PNGが欲しければ以下の行を有効化
    # plot_csv(series, os.path.join(OUT_DIR, "rbank9_intraday.png"))


if __name__ == "__main__":
    main()
