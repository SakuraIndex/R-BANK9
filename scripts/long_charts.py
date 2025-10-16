#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R-BANK9 charts (dark theme, level scale)
 - 背景: 黒ベース
 - 縦軸: level (指数の算出値)
 - 騰落率: (last - first) / first * 100 [%] に加え、Δlevelも出力
 - 変化率が異常に大きい場合は警告出力
"""

from pathlib import Path
import json
import math
import pandas as pd
import matplotlib.pyplot as plt

# ====== 設定 ======
INDEX_KEY = "rbank9"
OUTDIR = Path("docs/outputs")
CSV_INTRADAY = OUTDIR / f"{INDEX_KEY}_intraday.csv"
PNG_1D = OUTDIR / f"{INDEX_KEY}_1d.png"
TXT_POST = OUTDIR / f"{INDEX_KEY}_post_intraday.txt"
JSON_STATS = OUTDIR / f"{INDEX_KEY}_stats.json"
LEVEL_COL = "R_BANK9"
MARKET_TZ = "Asia/Tokyo"

# ====== Dark theme ======
plt.style.use("dark_background")
DARK_BG = "#0e0f13"
GRID = "#2a2e3a"
FG = "#e7ecf1"
GREEN = "#28e07c"
RED = "#ff4d4d"

def _apply_dark(ax, title: str, y_label: str):
    fig = ax.figure
    fig.set_facecolor(DARK_BG)
    ax.set_facecolor("#0b0c10")
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.tick_params(colors=FG)
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.set_title(title, color=FG, fontsize=12)
    ax.set_xlabel("Time", color=FG, fontsize=10)
    ax.set_ylabel(y_label, color=FG, fontsize=10)
    ax.grid(color=GRID, alpha=0.4)

# ====== データ処理 ======
def _read_df():
    df = pd.read_csv(CSV_INTRADAY)
    ts_col = df.columns[0]
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.set_index(ts_col).sort_index()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.index = df.index.tz_convert(MARKET_TZ)
    return df

def _calc_change(s: pd.Series):
    valid = s.replace([math.inf, -math.inf], pd.NA).dropna()
    valid = valid[valid > 0]
    if valid.empty:
        return None, None, None, "basis invalid=N/A"
    first, last = valid.iloc[0], valid.iloc[-1]
    pct = (last / first - 1) * 100.0
    diff = last - first
    note = "basis first-row invalid→used first valid"
    return pct, diff, (first, last), note

# ====== プロット ======
def _plot(df: pd.DataFrame, col: str, pct: float):
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    color = GREEN if pct is not None and pct >= 0 else RED
    _apply_dark(ax, f"{INDEX_KEY.upper()} (1d level)", "Index (level)")
    ax.plot(df.index, df[col], color=color, linewidth=1.8)
    fig.savefig(PNG_1D, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

# ====== 出力 ======
def _write_txt(pct, diff, note):
    if pct is None:
        line = f"{INDEX_KEY.upper()} 1d: N/A ({note})"
    else:
        warn = " ⚠️" if abs(pct) > 50 else ""
        sign = "+" if pct >= 0 else ""
        line = f"{INDEX_KEY.upper()} 1d: {sign}{pct:.2f}% Δ={diff:.4f} ({note}){warn}"
    TXT_POST.write_text(line + "\n", encoding="utf-8")

def _write_json(pct, diff):
    payload = {
        "index_key": INDEX_KEY,
        "pct_1d": None if pct is None else round(pct, 6),
        "delta_level": None if diff is None else round(diff, 6),
        "scale": "level",
        "updated_at": pd.Timestamp.utcnow().isoformat(timespec="seconds") + "Z",
    }
    JSON_STATS.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

# ====== Main ======
def main():
    df = _read_df()
    if LEVEL_COL not in df.columns:
        raise ValueError(f"missing column: {LEVEL_COL}")
    pct, diff, _, note = _calc_change(df[LEVEL_COL])
    _plot(df, LEVEL_COL, pct)
    _write_txt(pct, diff, note)
    _write_json(pct, diff)

if __name__ == "__main__":
    main()
