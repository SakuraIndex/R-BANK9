#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RBANK9 intraday/long chart generator (level scale)
- Open = first valid 'level'
- Close = last valid 'level'
- Color & 1d % = based purely on open vs close
- Dark theme
- Outliers winsorized for plotting only (not for stats)
- Writes:
    docs/outputs/rbank9_1d.png
    docs/outputs/rbank9_post_intraday.txt
    docs/outputs/rbank9_stats.json
"""

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Config ---------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "docs" / "outputs"
CSV_INTRADAY = OUT_DIR / "rbank9_intraday.csv"   # 既存CSV（ユーザーが提示した形式）
PNG_1D = OUT_DIR / "rbank9_1d.png"
TXT_POST = OUT_DIR / "rbank9_post_intraday.txt"
JSON_STATS = OUT_DIR / "rbank9_stats.json"

INDEX_KEY = "rbank9"
TITLE = "RBANK9 (1d level)"        # level（指数値）を縦軸に採用
DARK_BG = "#0B0F14"                # GitHubのダークと馴染む濃紺
GRID_COLOR = (1, 1, 1, 0.10)
AX_COLOR = (1, 1, 1, 0.85)
BULL_COLOR = "#00E676"             # 緑
BEAR_COLOR = "#FF5252"             # 赤
LINE_WIDTH = 2.2

# 外れ値処理（プロット専用）
WINSOR_SIGMA = 5.0

# ---------------------------------------- #

@dataclass
class DayEnds:
    open_level: float
    close_level: float
    pct_1d: float          # (close/open - 1) * 100
    delta_level: float     # close - open
    color: str             # plot color (bull or bear)


def _read_intraday(csv_path: Path) -> pd.DataFrame:
    """
    期待フォーマット：
        index: ISO8601 with timezone（例：2025-10-16 00:05:00+00:00）
        最終列: R_BANK9（= level、正の連続値）
    先頭数列は成分銘柄の寄与%が入るが、最後の列を指数レベルとして扱う。
    """
    df = pd.read_csv(csv_path, index_col=0)
    # インデックス → Datetime（tz-awareのまま利用）
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")

    # 最終列（右端）を指数レベルとみなす
    level = df.iloc[:, -1].astype(float)
    s = level.replace([np.inf, -np.inf], np.nan).dropna()
    return s.to_frame(name="level")


def _first_last_valid(s: pd.Series) -> Tuple[pd.Timestamp, float, pd.Timestamp, float]:
    """
    最初と最後の有効な level を返す。
    """
    first_idx = s.first_valid_index()
    last_idx = s.last_valid_index()
    if first_idx is None or last_idx is None:
        raise ValueError("No valid level values in intraday CSV.")
    return first_idx, float(s.loc[first_idx]), last_idx, float(s.loc[last_idx])


def _winsorize_for_plot(y: np.ndarray, sigma: float) -> np.ndarray:
    """
    見た目専用の外れ値抑制。中央値±sigma*MAD（近似）でクリップ。
    """
    if y.size == 0:
        return y
    med = np.nanmedian(y)
    mad = np.nanmedian(np.abs(y - med)) * 1.4826  # ≈ σ
    if not np.isfinite(mad) or mad == 0:
        return y
    lo = med - sigma * mad
    hi = med + sigma * mad
    return np.clip(y, lo, hi)


def _summarize(s: pd.Series) -> DayEnds:
    _, open_lv, _, close_lv = _first_last_valid(s)

    # 1day %
    pct_1d = (close_lv / open_lv - 1.0) * 100.0
    delta = close_lv - open_lv
    color = BULL_COLOR if close_lv >= open_lv else BEAR_COLOR

    return DayEnds(
        open_level=open_lv,
        close_level=close_lv,
        pct_1d=pct_1d,
        delta_level=delta,
        color=color,
    )


def _plot_level(s: pd.Series, ends: DayEnds) -> None:
    plt.close("all")
    plt.style.use("default")

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    # 軸・グリッドの色
    for spine in ax.spines.values():
        spine.set_color(AX_COLOR)
    ax.tick_params(colors=AX_COLOR)
    ax.yaxis.label.set_color(AX_COLOR)
    ax.xaxis.label.set_color(AX_COLOR)
    ax.title.set_color(AX_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=1.0)

    x = s.index
    y = s.values.astype(float)

    # 外れ値は見た目だけ winsorize
    y_plot = _winsorize_for_plot(y, WINSOR_SIGMA)

    ax.plot(x, y_plot, color=ends.color, linewidth=LINE_WIDTH)

    ax.set_title(TITLE)
    ax.set_xlabel("Time")
    ax.set_ylabel("Index (level)")

    # マージン（上下に少し余白）
    y_min = np.nanmin(y_plot)
    y_max = np.nanmax(y_plot)
    if np.isfinite(y_min) and np.isfinite(y_max) and y_max > y_min:
        pad = 0.05 * (y_max - y_min)
        ax.set_ylim(y_min - pad, y_max + pad)

    fig.tight_layout()
    PNG_1D.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_1D, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)


def _write_side_outputs(ends: DayEnds, basis_note: str) -> None:
    # テキスト（人間向け要約）
    sign = "+" if ends.pct_1d >= 0 else ""
    msg = f"RBANK9 1d: {sign}{ends.pct_1d:.2f}% Δ={ends.delta_level:+.4f} (basis {basis_note})"
    TXT_POST.write_text(msg + "\n", encoding="utf-8")

    # JSON（機械向け）
    JSON_STATS.write_text(
        json.dumps(
            {
                "index_key": INDEX_KEY,
                "pct_1d": round(ends.pct_1d, 6),
                "delta_level": round(ends.delta_level, 6),
                "scale": "level",  # ← 縦軸は level を明示
                "updated_at": pd.Timestamp.utcnow().isoformat().replace("+00:00", "Z"),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main() -> None:
    s_df = _read_intraday(CSV_INTRADAY)
    s = s_df["level"]

    # 最初の有効行を open、最後の有効行を close として固定
    first_ts, first_val, last_ts, last_val = _first_last_valid(s)

    # 基準時刻メモ：誤解回避のため出力文言に必ず記録
    basis_note = f"first-row valid={first_ts.isoformat()}→{last_ts.isoformat()}"

    # 要約（open/closeにのみ依存）
    ends = _summarize(s)

    # チャート描画（色はends.colorで決定）
    _plot_level(s, ends)

    # 付帯出力
    _write_side_outputs(ends, basis_note)


if __name__ == "__main__":
    main()
