#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
桜Index: intraday変化率を算出（PNGは触らない）
- intraday.csv を元に {key}_stats.json / {key}_post_intraday.txt を更新
- ％系列(pct)は前半/後半の中央値差でロバストに算出（外れ値・ゼロ始値対策）
"""

import os, json
from datetime import datetime, time as dtime
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd

OUTDIR = Path("docs/outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

INDEX_KEY = os.environ.get("INDEX_KEY", "index").lower()
MARKET_TZ = os.environ.get("MARKET_TZ", "Asia/Tokyo")
SESSION_START = os.environ.get("SESSION_START", "09:00")
SESSION_END   = os.environ.get("SESSION_END",   "15:30")

# 指数ごとのスケール固定
SCALE_MAP = {
    "scoin_plus": "price",     # 水準
    "rbank9": "pct",           # ％ポイント系列
    "ain10": "pct",            # ％ポイント系列
    "astra4": "fraction",      # 小数リターン系列（0.012 = 1.2%）
}

# ======== 読み込みユーティリティ ========
def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    df = pd.read_csv(path)
    tcols = [c for c in df.columns if c.lower() in ("datetime", "timestamp", "time", "日時", "date")]
    if not tcols:
        raise ValueError(f"{path} に日時列がありません")
    tcol = tcols[0]
    vcol = [c for c in df.columns if c.lower() not in ("datetime", "timestamp", "time", "日時", "date")][-1]

    df["ts_utc"] = pd.to_datetime(df[tcol], utc=True, errors="coerce")
    df["value"]  = pd.to_numeric(df[vcol], errors="coerce")
    df = df.dropna(subset=["ts_utc", "value"]).sort_values("ts_utc").reset_index(drop=True)
    return df[["ts_utc", "value"]]

def _to_market_tz(ts_utc: pd.Series) -> pd.Series:
    return ts_utc.dt.tz_convert(MARKET_TZ)

def _session_bounds(ts_local: pd.Series) -> Tuple[pd.Timestamp, pd.Timestamp]:
    if ts_local.empty:
        now_local = pd.Timestamp.utcnow()
        if now_local.tzinfo is None:
            now_local = now_local.tz_localize("UTC").tz_convert(MARKET_TZ)
        else:
            now_local = now_local.tz_convert(MARKET_TZ)
        base_date = now_local.date()
    else:
        base_date = ts_local.iloc[-1].date()

    s_h, s_m = map(int, SESSION_START.split(":"))
    e_h, e_m = map(int, SESSION_END.split(":"))
    start = pd.Timestamp(datetime.combine(base_date, dtime(s_h, s_m))).tz_localize(MARKET_TZ)
    end   = pd.Timestamp(datetime.combine(base_date, dtime(e_h, e_m))).tz_localize(MARKET_TZ)
    if end <= start:
        end += pd.Timedelta(days=1)
    return start, end

def _clamp_today(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    ts_local = _to_market_tz(df["ts_utc"])
    start, end = _session_bounds(ts_local)
    mask = (ts_local >= start) & (ts_local <= end)
    out = df.loc[mask].copy()
    if out.empty:
        last_day = ts_local.dt.date.iloc[-1]
        out = df.loc[ts_local.dt.date == last_day].copy()
    return out

# ======== 変化率の計算 ========
def _robust_open_close_pct(values: np.ndarray) -> Tuple[float, float]:
    """
    ％系列専用：前半/後半ブロックの中央値を open/close とみなす。
    - ゼロや極小値で平均が潰れる問題を避ける
    - 前半: 先頭から上位 max(5, n*0.1) 件
      後半: 末尾から上位 max(5, n*0.1) 件
    - 各ブロックは外れ値を除去（IQRの1.5倍ルール）
    """
    n = len(values)
    if n < 2:
        return np.nan, np.nan

    k = max(5, int(n * 0.1))
    head = values[:k]
    tail = values[-k:]

    def _rm_outliers(x: np.ndarray) -> np.ndarray:
        x = x[np.isfinite(x)]
        if len(x) == 0:
            return x
        q1, q3 = np.percentile(x, [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return x[(x >= lo) & (x <= hi)]

    head = _rm_outliers(head)
    tail = _rm_outliers(tail)

    if len(head) == 0 or len(tail) == 0:
        return np.nan, np.nan

    open_med  = float(np.median(head))
    close_med = float(np.median(tail))
    return open_med, close_med

def _compute_change(values: List[float], scale: str) -> float:
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return 0.0

    if scale == "pct":
        o, c = _robust_open_close_pct(arr)
        if np.isnan(o) or np.isnan(c):
            # フォールバック：単純な差分
            pct = arr[-1] - arr[0]
        else:
            pct = c - o
        # ガードレール：日中で±30ppを超えるのは異常値とみなしクリップ
        pct = float(np.clip(pct, -30.0, 30.0))
        return round(pct, 6)

    # price/fraction は従来どおり
    start, end = arr[0], arr[-1]

    # 極小始値→倍率暴発防止（price/fraction 共通）
    if abs(start) < 1e-6:
        start = float(np.median(arr[:max(5, len(arr)//10)]))

    if scale == "price":
        pct = (end / start - 1.0) * 100.0
    else:  # fraction
        pct = (end - start) * 100.0
    # price/fraction も常識的な範囲に丸める（±50%）
    pct = float(np.clip(pct, -50.0, 50.0))
    return round(pct, 6)

# ======== メイン ========
def main():
    df = _read_csv(OUTDIR / f"{INDEX_KEY}_intraday.csv")
    df = _clamp_today(df)

    scale = SCALE_MAP.get(INDEX_KEY, "pct")
    pct_1d = _compute_change(df["value"].tolist(), scale)

    stats = {
        "index_key": INDEX_KEY,
        "pct_1d": pct_1d,
        "scale": scale,
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }
    (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(json.dumps(stats, ensure_ascii=False), encoding="utf-8")

    sign = "+" if pct_1d >= 0 else ""
    (OUTDIR / f"{INDEX_KEY}_post_intraday.txt").write_text(
        f"{INDEX_KEY.upper()} 1d: {sign}{pct_1d:.2f}%",
        encoding="utf-8"
    )

    print(f"[{INDEX_KEY}] pct_1d={pct_1d:.3f} (scale={scale})")

if __name__ == "__main__":
    main()
