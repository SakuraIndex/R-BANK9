#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
桜Index: 日中騰落率の正規化出力（PNGは触らない版）
- intraday/history を読み、1d の変化率を % で算出
- {KEY}_stats.json と {KEY}_post_intraday.txt のみ更新（既存PNGは上書きしない）

ポイント
- 値スケールを自動判定（price / pct / fraction）
- 始値は「当日セッション内の最初の“非NaNかつ非ゼロ”」を採用（ダミー0.00を除外）
- history.csv の時刻列に 'date' を追加認識
- pandas 2.x でも安全な tz-localize / tz-convert に対応
"""

import os, json
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Tuple, List, Optional

import pandas as pd

# ========== 環境 ==========
OUTDIR = Path("docs/outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

INDEX_KEY = os.environ.get("INDEX_KEY", "index")
MARKET_TZ = os.environ.get("MARKET_TZ", "Asia/Tokyo")
SESSION_START = os.environ.get("SESSION_START", "09:00")
SESSION_END   = os.environ.get("SESSION_END",   "15:30")
CLAMP_SESSION = os.environ.get("CLAMP_SESSION", "true").lower() == "true"


# ========== ユーティリティ ==========
def _pick_value_column(df: pd.DataFrame) -> str:
    skip = {"time", "timestamp", "datetime", "日時", "date"}
    for col in reversed(df.columns):
        if col.lower() not in skip:
            return col
    raise ValueError("値列が見つかりませんでした")

def _read_csv_generic(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)

    time_cols = [c for c in df.columns if c.lower() in ("time", "timestamp", "datetime", "日時", "date")]
    if not time_cols:
        raise ValueError(f"{path} に time/timestamp/datetime/日時/date の列がありません")
    tcol = time_cols[0]

    df["ts_utc"] = pd.to_datetime(df[tcol], utc=True, errors="coerce")
    df = df.dropna(subset=["ts_utc"]).sort_values("ts_utc").reset_index(drop=True)

    vcol = _pick_value_column(df)
    df["value"] = pd.to_numeric(df[vcol], errors="coerce")
    df = df.dropna(subset=["value"])
    return df[["ts_utc", "value"]]

def _to_market_tz(ts_utc: pd.Series) -> pd.Series:
    return ts_utc.dt.tz_convert(MARKET_TZ)

def _session_bounds_for(ts_local: pd.Series):
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
    start = pd.Timestamp.combine(base_date, dtime(hour=s_h, minute=s_m), tz=MARKET_TZ)
    end   = pd.Timestamp.combine(base_date, dtime(hour=e_h, minute=e_m), tz=MARKET_TZ)
    if end <= start:
        end += pd.Timedelta(days=1)
    return start, end

def _clamp_session(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or not CLAMP_SESSION:
        return df
    ts_local = _to_market_tz(df["ts_utc"])
    start, end = _session_bounds_for(ts_local)
    mask = (ts_local >= start) & (ts_local <= end)
    out = df.loc[mask].copy()
    if out.empty:
        last_day = ts_local.dt.date.iloc[-1]
        out = df.loc[ts_local.dt.date == last_day].copy()
    return out

def _first_last_valid(vals: List[float]):
    # 始値: 最初の“非NaNかつ非ゼロ(極小を含む)”、終値: 最後の非NaN
    start = None
    for v in vals:
        if v is None:
            continue
        try:
            x = float(v)
        except Exception:
            continue
        if abs(x) > 1e-12:  # ダミー0.0を除外
            start = x
            break
    end = None
    for v in reversed(vals):
        if v is None:
            continue
        try:
            end = float(v)
            break
        except Exception:
            continue
    return start, end

def _detect_value_scale(s: float, e: float) -> str:
    max_abs = max(abs(s), abs(e))
    if max_abs > 20:
        return "price"
    elif max_abs < 1:
        return "fraction"   # 0.009 → 0.9%
    else:
        return "pct"        # -1.2 → -1.2%

def _compute_change_percent(vals: List[float]) -> Tuple[float, str]:
    s, e = _first_last_valid(vals)
    if s is None or e is None:
        return 0.0, "unknown"
    scale = _detect_value_scale(s, e)
    if scale == "price":
        chg = (e / s - 1.0) * 100.0
    elif scale == "pct":
        chg = (e - s)                # すでに％
    else:  # fraction
        chg = (e - s) * 100.0        # 小数 → ％
    return round(chg, 6), scale


# ========== メイン ==========
def main():
    intraday_csv = OUTDIR / f"{INDEX_KEY}_intraday.csv"
    history_csv  = OUTDIR / f"{INDEX_KEY}_history.csv"

    df_intraday = _read_csv_generic(intraday_csv) if intraday_csv.exists() else pd.DataFrame(columns=["ts_utc", "value"])
    df_history  = _read_csv_generic(history_csv)  if history_csv.exists()  else pd.DataFrame(columns=["ts_utc", "value"])

    # 当日セッション（intradayのみでOK）
    df_1d = _clamp_session(df_intraday.copy())

    pct_1d, scale_used = 0.0, "unknown"
    if not df_1d.empty:
        vals = df_1d["value"].tolist()
        pct_1d, scale_used = _compute_change_percent(vals)

    # 出力（PNGは触らない）
    stats = {
        "index_key": INDEX_KEY,
        "pct_1d": pct_1d,
        "scale": scale_used,
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }
    (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(json.dumps(stats, ensure_ascii=False), encoding="utf-8")

    sign = "+" if pct_1d >= 0 else ""
    (OUTDIR / f"{INDEX_KEY}_post_intraday.txt").write_text(
        f"{INDEX_KEY.upper()} 1d: {sign}{pct_1d:.2f}%",
        encoding="utf-8"
    )

    print(f"[{INDEX_KEY}] 1d change: {pct_1d:.3f}% (scale={scale_used})")
    print(f"[{INDEX_KEY}] wrote stats & marker only (kept existing PNGs).")

if __name__ == "__main__":
    main()
