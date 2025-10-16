#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RBANK9 intraday/long chart generator (level scale)
- Open = first valid 'level' (>EPS)
- Close = last valid 'level' (>EPS)
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
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Config ---------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "docs" / "outputs"
CSV_INTRADAY = OUT_DIR / "rbank9_intraday.csv"

PNG_1D = OUT_DIR / "rbank9_1d.png"
TXT_POST = OUT_DIR / "rbank9_post_intraday.txt"
JSON_STATS = OUT_DIR / "rbank9_stats.json"

INDEX_KEY = "rbank9"
TITLE = "RBANK9 (1d level)"        # 縦軸は level
DARK_BG = "#0B0F14"
GRID_COLOR = (1, 1, 1, 0.10)
AX_COLOR = (1, 1, 1, 0.85)
BULL_COLOR = "#00E676"             # 緑
BEAR_COLOR = "#FF5252"             # 赤
LINE_WIDTH = 2.2

# 見た目用ウィンザー化
WINSOR_SIGMA = 5.0

# level の最小許容値（ゼロ/負/極小は無効扱い）
EPS = 1e-9
# ---------------------------------------- #

@dataclass
class DayEnds:
    open_level: Optional[float]
    close_level: Optional[float]
    pct_1d: Optional[float]      # (close/open - 1) * 100
    delta_level: Optional[float] # close - open
    color: str                   # plot color


def _read_intraday(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    # 最終列を level とみなす（R_BANK9）
    level = pd.to_numeric(df.iloc[:, -1], errors="coerce")
    s = level.replace([np.inf, -np.inf], np.nan)
    return s.to_frame(name="level")


def _filter_valid_levels(s: pd.Series) -> pd.Series:
    """有効な level（finite かつ > EPS）のみ。"""
    return s[(np.isfinite(s)) & (s > EPS)]


def _first_last_valid(s: pd.Series) -> Tuple[Optional[pd.Timestamp], Optional[float],
                                             Optional[pd.Timestamp], Optional[float]]:
    if s.empty:
        return None, None, None, None
    first_idx = s.first_valid_index()
    last_idx = s.last_valid_index()
    if first_idx is None or last_idx is None:
        return None, None, None, None
    return first_idx, float(s.loc[first_idx]), last_idx, float(s.loc[last_idx])


def _winsorize_for_plot(y: np.ndarray, sigma: float) -> np.ndarray:
    if y.size == 0:
        return y
    med = np.nanmedian(y)
    mad = np.nanmedian(np.abs(y - med)) * 1.4826
    if not np.isfinite(mad) or mad == 0:
        return y
    lo = med - sigma * mad
    hi = med + sigma * mad
    return np.clip(y, lo, hi)


def _summarize(s_level: pd.Series) -> Tuple[DayEnds, str]:
    # 有効値のみで open/close を決定
    s_valid = _filter_valid_levels(s_level)
    f_ts, f_val, l_ts, l_val = _first_last_valid(s_valid)

    # 基準メモ
    if f_ts is None or l_ts is None:
        basis_note = "invalid=no valid level (>EPS)"
        return DayEnds(None, None, None, None, BEAR_COLOR), basis_note

    basis_note = f"first-row valid={f_ts.isoformat()}→{l_ts.isoformat()}"

    # 騰落率計算（open>EPS を再確認）
    if f_val is None or f_val <= EPS:
        basis_note += "; open<=EPS"
        return DayEnds(f_val, l_val, None, None, BEAR_COLOR), basis_note

    pct_1d = (l_val / f_val - 1.0) * 100.0
    delta = l_val - f_val
    color = BULL_COLOR if l_val >= f_val else BEAR_COLOR

    return DayEnds(f_val, l_val, pct_1d, delta, color), basis_note


def _plot_level(s: pd.Series, color: str) -> None:
    plt.close("all")
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    for spine in ax.spines.values():
        spine.set_color(AX_COLOR)
    ax.tick_params(colors=AX_COLOR)
    ax.yaxis.label.set_color(AX_COLOR)
    ax.xaxis.label.set_color(AX_COLOR)
    ax.title.set_color(AX_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=1.0)

    x = s.index
    y = s.values.astype(float)
    y_plot = _winsorize_for_plot(y, WINSOR_SIGMA)

    # データが全欠損でも安全
    if y_plot.size > 0:
        ax.plot(x, y_plot, color=color, linewidth=LINE_WIDTH)

        y_min = np.nanmin(y_plot)
        y_max = np.nanmax(y_plot)
        if np.isfinite(y_min) and np.isfinite(y_max) and y_max > y_min:
            pad = 0.05 * (y_max - y_min)
            ax.set_ylim(y_min - pad, y_max + pad)

    ax.set_title(TITLE)
    ax.set_xlabel("Time")
    ax.set_ylabel("Index (level)")

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_1D, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)


def _write_side_outputs(ends: DayEnds, basis_note: str) -> None:
    # テキスト（人向け）
    if ends.pct_1d is None:
        msg = f"RBANK9 1d: N/A (basis {basis_note})"
    else:
        sign = "+" if ends.pct_1d >= 0 else ""
        msg = f"RBANK9 1d: {sign}{ends.pct_1d:.2f}% Δ={ends.delta_level:+.4f} (basis {basis_note})"
    TXT_POST.write_text(msg + "\n", encoding="utf-8")

    # JSON（機械向け）
    payload = {
        "index_key": INDEX_KEY,
        "pct_1d": None if ends.pct_1d is None else round(ends.pct_1d, 6),
        "delta_level": None if ends.delta_level is None else round(ends.delta_level, 6),
        "scale": "level",
        "updated_at": pd.Timestamp.utcnow().isoformat().replace("+00:00", "Z"),
    }
    JSON_STATS.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    s_df = _read_intraday(CSV_INTRADAY)
    level_series = s_df["level"]

    ends, basis = _summarize(level_series)
    # プロット（色はends.color。N/Aでも生成可）
    _plot_level(level_series, ends.color)
    # サマリー出力
    _write_side_outputs(ends, basis)


if __name__ == "__main__":
    main()
