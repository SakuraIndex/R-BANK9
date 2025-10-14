#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
桜Index: 長期チャート生成 & 日中騰落率の正規化出力（全指数共通版）
- intraday/history を読み、PNG(1d/7d/1m/1y, intraday) を生成
- 値スケール自動判定（price / pct / fraction）で 1d 変化率を % で算出
- {KEY}_stats.json と {KEY}_post_intraday.txt を出力（サイト表示は stats.json を使用）
"""

import os
import json
from datetime import datetime, timedelta, time as dtime
from pathlib import Path
from typing import Tuple, List, Optional

import pandas as pd
import matplotlib.pyplot as plt


# ========== 環境 ==========
OUTDIR = Path("docs/outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

INDEX_KEY = os.environ.get("INDEX_KEY", "index")          # 例: "rbank9" / "ain10" / "scoin_plus" / "astra4"
MARKET_TZ = os.environ.get("MARKET_TZ", "Asia/Tokyo")      # 例: "Asia/Tokyo", "America/New_York", "UTC"
SESSION_START = os.environ.get("SESSION_START", "09:00")   # "HH:MM"
SESSION_END   = os.environ.get("SESSION_END",   "15:30")   # "HH:MM"
CLAMP_SESSION = os.environ.get("CLAMP_SESSION", "true").lower() == "true"

PNG_LINEWIDTH = 1.5


# ========== ユーティリティ ==========
def _pick_value_column(df: pd.DataFrame) -> str:
    """最右列（time/timestamp/datetime/日時 以外）を値列とみなす"""
    candidates = [c for c in df.columns if c.lower() not in ("time", "timestamp", "datetime", "日時")]
    if not candidates:
        raise ValueError("値列が見つかりませんでした")
    return candidates[-1]


def _read_csv_generic(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    # 時刻列の正規化
    time_cols = [c for c in df.columns if c.lower() in ("time", "timestamp", "datetime", "日時")]
    if not time_cols:
        raise ValueError(f"{path} に time/timestamp/datetime/日時 の列がありません")
    tcol = time_cols[0]
    df["ts_utc"] = pd.to_datetime(df[tcol], utc=True, errors="coerce")
    df = df.dropna(subset=["ts_utc"]).sort_values("ts_utc").reset_index(drop=True)
    # 値列の抽出
    vcol = _pick_value_column(df)
    df["value"] = pd.to_numeric(df[vcol], errors="coerce")
    df = df.dropna(subset=["value"])
    return df[["ts_utc", "value"]]


def _detect_value_scale(values: List[float]) -> str:
    """値のスケールを自動判定: 'price' / 'pct' / 'fraction'"""
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return "unknown"
    s, e = vals[0], vals[-1]
    max_abs = max(abs(s), abs(e))
    if max_abs > 20:
        return "price"      # 例: 100, 101
    elif max_abs < 1:
        return "fraction"   # 例: 0.004, -0.012 （= 0.4%, -1.2%）
    else:
        return "pct"        # 例: -1.2, +0.5 （= 既に%単位）


def _compute_change_percent(values: List[float]) -> Tuple[float, str]:
    """スケール自動判定のうえ、常に '%' で変化率を返す"""
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return 0.0, "unknown"
    s, e = vals[0], vals[-1]
    scale = _detect_value_scale(vals)
    if scale == "price":
        chg = (e / s - 1.0) * 100.0 if s != 0 else 0.0
    elif scale == "pct":
        chg = (e - s)  # 既に% → 差分（パーセントポイント）
    elif scale == "fraction":
        chg = (e - s) * 100.0  # 小数 → % へ
    else:
        chg = 0.0
    return round(chg, 6), scale


def _to_market_tz(ts_utc: pd.Series) -> pd.Series:
    return ts_utc.dt.tz_convert(MARKET_TZ)


def _session_bounds_for(ts_local: pd.Series) -> Tuple[datetime, datetime]:
    """最新営業日のセッション開始・終了のローカル時刻（マーケットTZ）を返す"""
    if ts_local.empty:
        now_local = pd.Timestamp.utcnow().tz_localize("UTC").tz_convert(MARKET_TZ)
        base_date = now_local.date()
    else:
        base_date = ts_local.iloc[-1].date()

    s_h, s_m = map(int, SESSION_START.split(":"))
    e_h, e_m = map(int, SESSION_END.split(":"))
    start = pd.Timestamp(datetime.combine(base_date, dtime(hour=s_h, minute=s_m)), tz=MARKET_TZ)
    end   = pd.Timestamp(datetime.combine(base_date, dtime(hour=e_h, minute=e_m)), tz=MARKET_TZ)
    if end <= start:
        end += pd.Timedelta(days=1)
    return (start.to_pydatetime(), end.to_pydatetime())


def _clamp_session(df: pd.DataFrame) -> pd.DataFrame:
    """マーケットTZの当日セッション時間帯で絞り込む（CLAMP_SESSION=true の時）"""
    if df.empty or not CLAMP_SESSION:
        return df
    ts_local = _to_market_tz(df["ts_utc"])
    start, end = _session_bounds_for(ts_local)
    mask = (ts_local >= start) & (ts_local <= end)
    out = df.loc[mask].copy()
    if out.empty:
        # もし当日セッションにデータがなければ直近日の全データを返す
        last_day = ts_local.dt.date.iloc[-1]
        out = df.loc[ts_local.dt.date == last_day].copy()
    return out


def _plot_series(df: pd.DataFrame, title: str, outpath: Path):
    if df.empty:
        return
    ts_local = _to_market_tz(df["ts_utc"])
    plt.figure(figsize=(6.5, 3.2), dpi=160)
    plt.plot(ts_local, df["value"], linewidth=PNG_LINEWIDTH)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Index / Value")
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath)
    plt.close()


def _window_from_history(df_hist: pd.DataFrame, days: int) -> pd.DataFrame:
    if df_hist is None or df_hist.empty:
        return pd.DataFrame(columns=["ts_utc", "value"])
    cutoff = pd.Timestamp.utcnow().tz_localize("UTC") - pd.Timedelta(days=days)
    return df_hist.loc[df_hist["ts_utc"] >= cutoff].copy()


# ========== メイン処理 ==========
def main():
    # 1) 入力ファイル検出
    intraday_csv = OUTDIR / f"{INDEX_KEY}_intraday.csv"
    history_csv  = OUTDIR / f"{INDEX_KEY}_history.csv"

    df_intraday = _read_csv_generic(intraday_csv) if intraday_csv.exists() else pd.DataFrame(columns=["ts_utc", "value"])
    df_history  = _read_csv_generic(history_csv)  if history_csv.exists()  else pd.DataFrame(columns=["ts_utc", "value"])

    # 2) セッション内 1D ウィンドウ（intraday）
    df_1d_raw = df_intraday.copy()
    df_1d = _clamp_session(df_1d_raw)

    # 3) 7d/1m/1y ウィンドウ（history ベース / fallback intraday）
    df_7d = _window_from_history(df_history if not df_history.empty else df_intraday, 7)
    df_1m = _window_from_history(df_history if not df_history.empty else df_intraday, 31)
    df_1y = _window_from_history(df_history if not df_history.empty else df_intraday, 366)

    # 4) 画像生成
    _plot_series(df_1d, f"{INDEX_KEY.upper()} (1d)", OUTDIR / f"{INDEX_KEY}_1d.png")
    _plot_series(df_7d, f"{INDEX_KEY.upper()} (7d)", OUTDIR / f"{INDEX_KEY}_7d.png")
    _plot_series(df_1m, f"{INDEX_KEY.upper()} (1m)", OUTDIR / f"{INDEX_KEY}_1m.png")
    _plot_series(df_1y, f"{INDEX_KEY.upper()} (1y)", OUTDIR / f"{INDEX_KEY}_1y.png")
    _plot_series(df_intraday, f"{INDEX_KEY.upper()} (intraday)", OUTDIR / f"{INDEX_KEY}_intraday.png")

    # 5) 1d 変化率（％）を自動判定で算出
    pct_1d = 0.0
    scale_used = "unknown"
    if not df_1d.empty:
        vals = df_1d["value"].tolist()
        pct_1d, scale_used = _compute_change_percent(vals)

    # 6) stats.json / post_intraday.txt を出力
    stats = {
        "pct_1d": pct_1d,
        "scale": scale_used,                 # 監査用
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "index_key": INDEX_KEY,
    }
    (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(json.dumps(stats, ensure_ascii=False), encoding="utf-8")

    sign = "+" if pct_1d >= 0 else ""
    (OUTDIR / f"{INDEX_KEY}_post_intraday.txt").write_text(
        f"{INDEX_KEY.upper()} 1d: {sign}{pct_1d:.2f}%",
        encoding="utf-8"
    )

    # 7) ログ
    print(f"[{INDEX_KEY}] 1d change: {pct_1d:.3f}% (scale={scale_used})")
    print(f"[{INDEX_KEY}] outputs -> {OUTDIR}")


if __name__ == "__main__":
    main()
