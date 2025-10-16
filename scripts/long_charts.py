#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R-BANK9 charts + stats
- ソース: docs/outputs/rbank9_intraday.csv（UTCタイムスタンプ）
- 仕様: 構成銘柄の 1銘柄あたり騰落率(%) を行方向に平均して「指数の％」を作る
- 表示: デフォルトは「前日終値比（％）」; 必要なら寄り付き比にも切替可
- テーマ: ダーク / 終値の符号で線色自動切替
"""

from __future__ import annotations
from pathlib import Path
import json
from datetime import datetime, timezone
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------
# constants / paths
# ------------------------
INDEX_KEY = "rbank9"
OUTDIR = Path("docs/outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

HISTORY_CSV  = OUTDIR / f"{INDEX_KEY}_history.csv"   # 未使用でも残す
INTRADAY_CSV = OUTDIR / f"{INDEX_KEY}_intraday.csv"

# 「％の基準」
#   "prev_close" : 前日終値比（％） … row 平均そのものを表示
#   "from_open"  : 当日寄り付き比（％） … row 平均 － 寄り付き時点の平均
MODE = "prev_close"   # 必要に応じて "from_open" に変更可

# ------------------------
# plotting style (dark)
# ------------------------
DARK_BG = "#0e0f13"
DARK_AX = "#0b0c10"
FG_TEXT = "#e7ecf1"
GRID    = "#2a2e3a"
GREEN   = "#28e07c"   # 上昇
RED     = "#ff4d4d"   # 下落
NEUTRAL = "#d5d8de"

def _apply(ax, title: str, ylabel: str = "Change (%)") -> None:
    fig = ax.figure
    fig.set_size_inches(12, 7)
    fig.set_dpi(160)
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_AX)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.grid(color=GRID, alpha=0.6, linewidth=0.8)
    ax.tick_params(colors=FG_TEXT, labelsize=10)
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.set_title(title, color=FG_TEXT, fontsize=12)
    ax.set_xlabel("Time", color=FG_TEXT, fontsize=10)
    ax.set_ylabel(ylabel, color=FG_TEXT, fontsize=10)

def _save(x: pd.Index, y: pd.Series, out_png: Path, title: str) -> None:
    """終値の符号で線色切替。データが薄い場合は中立色。"""
    if len(y) == 0 or not np.isfinite(y.iloc[-1]):
        color = NEUTRAL
    else:
        color = GREEN if y.iloc[-1] >= 0 else RED

    fig, ax = plt.subplots()
    _apply(ax, title)
    ax.plot(x, y, color=color, linewidth=1.6)
    fig.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

# ------------------------
# loading & shaping
# ------------------------
def _load_intraday_pct() -> Tuple[pd.DatetimeIndex, pd.Series]:
    """
    CSV から「各構成銘柄の騰落率(%)」列を抽出し、行平均（％）の Series を返す。
    先頭行のゼロ埋め(プレ開場)や欠損のみの行は除外。
    """
    if not INTRADAY_CSV.exists():
        raise FileNotFoundError(f"{INTRADAY_CSV} not found.")

    df = pd.read_csv(INTRADAY_CSV, parse_dates=[0], index_col=0)
    # 列名の正規化
    cols = [c.strip() for c in df.columns]
    df.columns = cols

    # 構成銘柄列 = R_BANK9 以外の全て
    member_cols = [c for c in df.columns if c.upper() != "R_BANK9"]

    # 数値化（％の実数）、非数は NaN
    for c in member_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 「プレ開場（ゼロしかない/有効列が極少）」な行を落とす
    valid_count = df[member_cols].notna().sum(axis=1)
    nonzero_sum = (df[member_cols].fillna(0).abs() > 0).sum(axis=1)
    mask_valid = (valid_count >= max(3, int(len(member_cols) * 0.4))) & (nonzero_sum >= 1)
    df = df.loc[mask_valid].copy()

    if df.empty:
        # データが薄すぎる場合は空を返す
        return df.index, pd.Series(dtype=float)

    # 行平均（％）
    row_mean_pct = df[member_cols].mean(axis=1, skipna=True)

    # 表示モード: 前日終値比 or 当日寄り付き比
    if MODE == "from_open":
        # 当日の最初の有効値を寄り付きとみなす
        base = row_mean_pct.iloc[0]
        y = row_mean_pct - base
        suffix = " (1d %, vs open)"
    else:
        y = row_mean_pct
        suffix = " (1d %, vs prev close)"

    # タイムスタンプ & 出力タイトル用サフィックス
    y.name = f"{INDEX_KEY.upper()}{suffix}"
    return df.index, y

# ------------------------
# charts
# ------------------------
def gen_pngs() -> float | None:
    """
    PNG を生成し、最終値（％）を返す（サイト表示用）。
    """
    x, y = _load_intraday_pct()

    # 1d は当日分すべて
    if len(y) == 0:
        last_pct = None
    else:
        last_pct = float(y.iloc[-1])

    # 長期も同一ロジックで（現状は intraday のみを描画）
    title1d = f"{INDEX_KEY.upper()} (1d %)"
    _save(x, y, OUTDIR / f"{INDEX_KEY}_1d.png", title1d)

    # 7d/1m/1y は当面、これと同フォーマットで出す（データがあれば将来拡張）
    # ここでは 1d と同じものを保存しておく（サイト側の参照を満たすため）
    _save(x, y, OUTDIR / f"{INDEX_KEY}_7d.png", f"{INDEX_KEY.upper()} (7d %)")
    _save(x, y, OUTDIR / f"{INDEX_KEY}_1m.png", f"{INDEX_KEY.upper()} (1m %)")
    _save(x, y, OUTDIR / f"{INDEX_KEY}_1y.png", f"{INDEX_KEY.upper()} (1y %)")

    return last_pct

# ------------------------
# stats + marker
# ------------------------
def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def write_stats_and_marker(pct_last: float | None) -> None:
    payload = {
        "index_key": INDEX_KEY,
        "pct_1d": None if pct_last is None or not np.isfinite(pct_last) else round(float(pct_last), 6),
        "scale": "pct",   # サイト側 “pct” で解釈
        "updated_at": _now_utc_iso(),
    }
    (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )

    marker = OUTDIR / f"{INDEX_KEY}_post_intraday.txt"
    if payload["pct_1d"] is None:
        marker.write_text(f"{INDEX_KEY.upper()} 1d: N/A\n", encoding="utf-8")
    else:
        marker.write_text(f"{INDEX_KEY.upper()} 1d: {payload['pct_1d']:+.2f}%\n", encoding="utf-8")

# ------------------------
# main
# ------------------------
if __name__ == "__main__":
    last = gen_pngs()
    write_stats_and_marker(last)
