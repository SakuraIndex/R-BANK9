#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R-BANK9 charts + stats
 - CSVの時刻がUTCでもJSTでも自動判定
 - JST場中(09:00–15:30)のみ使用し、％算出
 - ロバスト基準値（最初の15分中央値）
 - ダークテーマ / 色は終値の符号
"""
from pathlib import Path
from datetime import datetime, time, timedelta, timezone
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INDEX_KEY = "rbank9"
OUTDIR = Path("docs/outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)
INTRADAY_CSV = OUTDIR / f"{INDEX_KEY}_intraday.csv"
HISTORY_CSV = OUTDIR / f"{INDEX_KEY}_history.csv"

MARKET_TZ = "Asia/Tokyo"
SESSION_S = time(9, 0)
SESSION_E = time(15, 30)
BASE_WINDOW_MIN = 15

# dark theme
DARK_BG = "#0e0f13"; DARK_AX = "#0b0c10"; FG_TEXT = "#e7ecf1"
GRID = "#2a2e3a"; GREEN = "#28e07c"; RED = "#ff4d4d"

def _apply(ax, title, ylabel):
    fig = ax.figure
    fig.set_size_inches(12, 7); fig.set_dpi(160)
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_AX)
    for s in ax.spines.values(): s.set_color(GRID)
    ax.grid(color=GRID, alpha=0.6, linewidth=0.8)
    ax.tick_params(colors=FG_TEXT)
    ax.set_title(title, color=FG_TEXT)
    ax.set_xlabel("Time", color=FG_TEXT)
    ax.set_ylabel(ylabel, color=FG_TEXT)

def _detect_tz(df):
    """CSVがUTC or JSTかを自動判定（時刻平均が9〜15時に集中してたらJSTとみなす）"""
    hours = df.index.hour
    jst_like = ((hours >= 9) & (hours <= 15)).mean() > 0.5
    return "JST" if jst_like else "UTC"

def _load_df():
    f = INTRADAY_CSV if INTRADAY_CSV.exists() else HISTORY_CSV
    if not f.exists():
        raise FileNotFoundError("No CSV found")
    df = pd.read_csv(f, parse_dates=[0], index_col=0)
    df.columns = [c.strip().lower() for c in df.columns]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(how="all")
    return df

def _to_jst(df):
    if df.empty: return df
    # タイムゾーン判定
    tztype = _detect_tz(df)
    if tztype == "UTC":
        df.index = df.index.tz_localize("UTC").tz_convert(MARKET_TZ)
    else:
        # 既にJSTなら tz_localize
        df.index = df.index.tz_localize(MARKET_TZ)
    return df

def _slice_today_jst(df):
    if df.empty: return df
    last_day = df.index[-1].date()
    start = datetime.combine(last_day, SESSION_S).astimezone(df.index.tz)
    end   = datetime.combine(last_day, SESSION_E).astimezone(df.index.tz)
    mask = (df.index >= start) & (df.index <= end)
    day_df = df.loc[mask]
    return day_df

def _robust_base(s):
    if s.empty: return None
    t0 = s.index[0]; t1 = t0 + timedelta(minutes=BASE_WINDOW_MIN)
    v = s[(s.index >= t0) & (s.index <= t1)].dropna()
    v = v[v > 0]
    return float(np.median(v)) if len(v) >= 2 else None

def _save_line(x, y, out, title, ylabel, color):
    fig, ax = plt.subplots()
    _apply(ax, title, ylabel)
    ax.plot(x, y, color=color, linewidth=1.6)
    fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

def main():
    df = _load_df()
    col = df.columns[-1]
    df = _to_jst(df)
    day = _slice_today_jst(df)
    pct_last = None; base=None; last=None

    if not day.empty:
        base = _robust_base(day[col])
        last = float(day[col].iloc[-1])
        if base and last:
            pct = (day[col]/base - 1)*100
            pct_last = float(pct.iloc[-1])
            color = GREEN if pct_last >= 0 else RED
            _save_line(pct.index, pct, OUTDIR/f"{INDEX_KEY}_1d.png", f"{INDEX_KEY.upper()} (1d %)", "Change (%)", color)
        else:
            _save_line(day.index, day[col]*0, OUTDIR/f"{INDEX_KEY}_1d.png", f"{INDEX_KEY.upper()} (1d %)", "Change (%)", FG_TEXT)
    else:
        _save_line(df.index, df[col]*0, OUTDIR/f"{INDEX_KEY}_1d.png", f"{INDEX_KEY.upper()} (1d %)", "Change (%)", FG_TEXT)

    # stats + marker
    payload = {
        "index_key": INDEX_KEY,
        "pct_1d": None if pct_last is None or np.isnan(pct_last) else round(pct_last, 6),
        "scale": "pct",
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00","Z"),
    }
    (OUTDIR/f"{INDEX_KEY}_stats.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    marker = OUTDIR/f"{INDEX_KEY}_post_intraday.txt"
    if pct_last is None:
        marker.write_text(f"{INDEX_KEY.upper()} 1d: N/A\n", encoding="utf-8")
    else:
        marker.write_text(f"{INDEX_KEY.upper()} 1d: {pct_last:+.2f}% (base={base}, last={last})\n", encoding="utf-8")

if __name__ == "__main__":
    main()
