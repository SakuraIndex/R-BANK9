# scripts/long_charts.py
# -*- coding: utf-8 -*-
"""
RBANK9 長期チャート/サマリ生成（1d/7d/1m/1y）:
- intraday(1d) は「レベル」系列（％ポイント）を扱うため、1日の変化量は“差(Δ)”で評価する
- 終値/始値の割り算で騰落率を出さない（誤りの原因）
- 背景はダーク、グリッド無し、線色は Δ の符号で決定
- サマリは scale='level' とし、pct_1d は None、delta_level を出力
- 基準は CSV の最初の有効行（first valid）。無効/ゼロ/NaN は飛ばす
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import matplotlib.pyplot as plt

# ====== 設定 ======
ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "outputs"

INTRADAY_CSV = DOCS / "rbank9_intraday.csv"
HISTORY_CSV  = DOCS / "rbank9_history.csv"   # 長期で使っていれば
IMG_1D       = DOCS / "rbank9_1d.png"
IMG_7D       = DOCS / "rbank9_7d.png"
IMG_1M       = DOCS / "rbank9_1m.png"
IMG_1Y       = DOCS / "rbank9_1y.png"
POST_TXT     = DOCS / "rbank9_post_intraday.txt"
STATS_JSON   = DOCS / "rbank9_stats.json"

INDEX_KEY = "rbank9"      # 統計キー
COL_LEVEL  = "R_BANK9"    # intraday の合成列名（末尾列）

# ====== ユーティリティ ======

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _coerce_float(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s

def _first_last_valid(s: pd.Series):
    """最初と最後の有効値（finite）とそれぞれの時刻を返す。なければ (None, None, None, None)"""
    valid = s.dropna()
    # finite のみ
    valid = valid[~(pd.isna(valid) | ~pd.Series(pd.to_numeric(valid, errors="coerce")).apply(lambda x: math.isfinite(x)))]
    if valid.empty:
        return None, None, None, None
    first_ts = valid.index[0]
    last_ts  = valid.index[-1]
    return float(valid.iloc[0]), float(valid.iloc[-1]), first_ts, last_ts

def _format_pp(x: float | None) -> str:
    if x is None or not math.isfinite(x):
        return "N/A"
    # ％ポイント表示（pp）。ユーザー表示は ±0.00% としても意味は「pp」
    return f"{x:+.2f}%"

def _choose_color(delta_level: float | None) -> str:
    if delta_level is None or not math.isfinite(delta_level):
        return "#9AA0A6"  # グレー（無判定）
    return "#2ECC71" if delta_level >= 0 else "#FF4D4D"  # 緑 / 赤

def _apply_dark(ax: plt.Axes):
    """ダーク背景 + グリッド無し + 軸・文字色統一"""
    bg = "#0f1115"
    fg = "#d0d4dc"
    ax.figure.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    for spine in ax.spines.values():
        spine.set_color("#3b4048")
    ax.tick_params(colors=fg, labelsize=10)
    ax.xaxis.label.set_color(fg)
    ax.yaxis.label.set_color(fg)
    ax.title.set_color(fg)
    ax.grid(False)

# ====== サマリ（1d） ======

def summarize_intraday_level() -> dict:
    """
    intraday(1d) 用サマリ。
    - scale は 'level'（％ポイントのレベル）
    - pct_1d は None（誤解を避けるため出さない）
    - delta_level = close_lv - open_lv（％ポイント差）
    """
    df = pd.read_csv(INTRADAY_CSV)
    # index to datetime (tz-aware)
    idx = pd.to_datetime(df.iloc[:, 0], utc=True, errors="coerce")
    s = _coerce_float(df[COL_LEVEL])
    s.index = idx

    open_lv, close_lv, t_open, t_close = _first_last_valid(s)
    result = {
        "index_key": INDEX_KEY,
        "pct_1d": None,             # ここは None（レベルに対して%は出さない）
        "delta_level": None,         # ％ポイント差
        "scale": "level",
        "updated_at": _now_iso(),
    }

    if open_lv is None or close_lv is None:
        POST_TXT.write_text("RBANK9 1d: N/A (basis invalid-N/A)\n", encoding="utf-8")
        STATS_JSON.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        return result

    delta_lv = close_lv - open_lv  # ％ポイント差（レベルはすでに % 表示の平均）
    result["delta_level"] = round(delta_lv, 5)

    basis_note = f"(basis first-row valid={t_open.isoformat()}–{t_close.isoformat()})"
    POST_TXT.write_text(
        f"RBANK9 1d: Δ={_format_pp(delta_lv)} {basis_note}\n",
        encoding="utf-8"
    )
    STATS_JSON.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result

# ====== 描画 ======

def plot_intraday_level():
    df = pd.read_csv(INTRADAY_CSV)
    idx = pd.to_datetime(df.iloc[:, 0], utc=True, errors="coerce")
    y = _coerce_float(df[COL_LEVEL])
    y.index = idx

    # 線色は Δ の符号で
    open_lv, close_lv, _, _ = _first_last_valid(y)
    delta_lv = (close_lv - open_lv) if (open_lv is not None and close_lv is not None) else None
    color = _choose_color(delta_lv)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    _apply_dark(ax)

    ax.plot(y.index, y.values, color=color, linewidth=2)
    ax.set_title("RBANK9 (1d level)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Index (level)")  # 目盛りは %（平均％）だが、単位名は level のままにする

    # 余白を少し
    ax.margins(x=0.02, y=0.10)
    fig.tight_layout()
    fig.savefig(IMG_1D, facecolor=fig.get_facecolor(), dpi=160)
    plt.close(fig)

# --- 7d/1m/1y は従来ロジック（必要ならここで同様のスタイルを適用） ---
# 既存の長期 CSV/生成処理が別にある前提。ここでは intraday(1d) の不整合修正に集中。

# ====== エントリーポイント ======

def main():
    summarize_intraday_level()
    plot_intraday_level()

if __name__ == "__main__":
    main()
