#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R-BANK9 charts + stats  (level → pct, robust baseline, auto line color, dark theme)
"""

from pathlib import Path
import json
from datetime import datetime, timezone, timedelta
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------
# constants / paths
# ------------------------
INDEX_KEY = "rbank9"
OUTDIR = Path("docs/outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

HISTORY_CSV  = OUTDIR / f"{INDEX_KEY}_history.csv"
INTRADAY_CSV = OUTDIR / f"{INDEX_KEY}_intraday.csv"

# ▼ R-BANK9 は“指数レベル”として固定（ヒューリスティック無効化）
SCALE_MODE = "level"   # "level" 固定

# ------------------------
# plotting style (dark)
# ------------------------
DARK_BG = "#0e0f13"
DARK_AX = "#0b0c10"
FG_TEXT = "#e7ecf1"
GRID    = "#2a2e3a"
GREEN   = "#28e07c"   # 上昇
RED     = "#ff4d4d"   # 下落

def _apply(ax, title: str, y_label: str) -> None:
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
    ax.set_ylabel(y_label, color=FG_TEXT, fontsize=10)

def _save_line(x, y, out_png: Path, title: str, color: str, y_label: str) -> None:
    fig, ax = plt.subplots()
    _apply(ax, title, y_label)
    ax.plot(x, y, color=color, linewidth=1.6)
    fig.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

# ------------------------
# data loading helpers
# ------------------------
def _pick_index_column(df: pd.DataFrame) -> str:
    """
    R-BANK9 列を推定（既知別名を許容）。無ければ最後の列。
    """
    cand_names = {
        "rbank9", "r_bank9", "rbnk9", "rbank_9", "r_bank_9", "r-bank9",
        INDEX_KEY, INDEX_KEY.upper(), "R_BANK9", "RBANK9"
    }
    for c in df.columns:
        if c and c.strip().lower() in cand_names:
            return c
    return df.columns[-1]

def _load_df() -> pd.DataFrame:
    """
    intraday があれば intraday 優先、無ければ history。
    インデックスは tz-aware (UTC) を維持。数値列に強制変換。
    """
    if INTRADAY_CSV.exists():
        df = pd.read_csv(INTRADAY_CSV, parse_dates=[0], index_col=0)
    elif HISTORY_CSV.exists():
        df = pd.read_csv(HISTORY_CSV, parse_dates=[0], index_col=0)
    else:
        raise FileNotFoundError("R-BANK9: neither intraday nor history csv found.")
    for c in list(df.columns):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(how="all")
    return df

def _slice_today_utc(df: pd.DataFrame) -> pd.DataFrame:
    """
    直近24時間 (UTC) を切り出し。R-BANK9 は日本株だが、CSVはUTCで来ている想定。
    """
    if len(df.index) == 0:
        return df
    last_ts = df.index[-1]
    # tz-naive でも tz-aware(UTC) でも動くように
    if last_ts.tzinfo is None:
        end = pd.Timestamp(last_ts).tz_localize("UTC")
    else:
        end = last_ts.tz_convert("UTC")
    start = end - pd.Timedelta(days=1)
    return df.loc[(df.index >= start) & (df.index <= end)]

def _robust_baseline(series: pd.Series) -> float | None:
    """
    最初の 3〜6 点程度から外れ値を除去し、中央値でベースを決定。
    0 以下・NaN は無効として None を返す。
    """
    s = series.dropna()
    if len(s) == 0:
        return None
    head = s.iloc[: max(3, min(6, len(s)))]   # 最初の3〜6点
    # 外れ値除去: IQR で緩めに
    q1, q3 = head.quantile([0.25, 0.75])
    iqr = float(q3 - q1) if pd.notna(q3) and pd.notna(q1) else 0.0
    low, high = (q1 - 3 * iqr, q3 + 3 * iqr) if iqr > 0 else (head.min(), head.max())
    trimmed = head[(head >= low) & (head <= high)]
    base = float(trimmed.median()) if len(trimmed) else float(head.median())
    if not math.isfinite(base) or base <= 0:
        return None
    return base

# ------------------------
# chart generation
# ------------------------
def _make_pct_series_from_level(level: pd.Series) -> pd.Series:
    """
    レベル → 当日比 (%) に変換。ベースが取れない場合は NaN を返す。
    """
    level = level.dropna()
    if len(level) < 1:
        return pd.Series(dtype=float, index=level.index)
    base = _robust_baseline(level)
    if base is None or base <= 0:
        # ベースが取れない時は NaN 連番
        return pd.Series([np.nan] * len(level), index=level.index, dtype=float)
    pct = (level / base - 1.0) * 100.0
    return pct.astype(float)

def gen_pngs_and_stats() -> None:
    df_all = _load_df()
    col = _pick_index_column(df_all)

    # 当日(直近24h UTC)に絞る
    df = _slice_today_utc(df_all)
    s = df[col].astype(float) if col in df.columns else df.iloc[:, -1].astype(float)

    # ── レベル → % 変換（固定）
    pct_1d_series = _make_pct_series_from_level(s)

    # ライン色: 終値の符号
    last_change = pct_1d_series.dropna().iloc[-1] if len(pct_1d_series.dropna()) else np.nan
    color = GREEN if (pd.notna(last_change) and last_change >= 0) else RED

    # 1d/7d/1m/1y は、R-BANK9 に関しては「%」表記の1dのみ更新するのが安全。
    # （長期の % は参考値になりづらいので、ここでは 1d のみ % にし、他はレベルで描きたい場合は別PNGを足す）
    # まず 1d（%）
    _save_line(
        pct_1d_series.index, pct_1d_series.values,
        OUTDIR / f"{INDEX_KEY}_1d.png",
        f"{INDEX_KEY.upper()} (1d %)",
        color,
        "Change (%)"
    )

    # 参考: レベル系列の 7d/1m/1y（色は 1d と合わせる）
    # 7d
    s_7d = df_all[col].astype(float).dropna().iloc[-7*24*60:] if len(df_all) else pd.Series(dtype=float)
    _save_line(s_7d.index, s_7d.values, OUTDIR / f"{INDEX_KEY}_7d.png", f"{INDEX_KEY.upper()} (7d)", color, "Index / Value")
    # 1m
    s_1m = df_all[col].astype(float).dropna().iloc[-30*24*60:] if len(df_all) else pd.Series(dtype=float)
    _save_line(s_1m.index, s_1m.values, OUTDIR / f"{INDEX_KEY}_1m.png", f"{INDEX_KEY.upper()} (1m)", color, "Index / Value")
    # 1y
    s_1y = df_all[col].astype(float).dropna() if len(df_all) else pd.Series(dtype=float)
    _save_line(s_1y.index, s_1y.values, OUTDIR / f"{INDEX_KEY}_1y.png", f"{INDEX_KEY.upper()} (1y)", color, "Index / Value")

    # ── stats.json & human marker
    if len(pct_1d_series.dropna()) == 0:
        pct_val = None
    else:
        pct_val = float(pct_1d_series.dropna().iloc[-1])

    payload = {
        "index_key": INDEX_KEY,
        "pct_1d": None if pct_val is None or not math.isfinite(pct_val) else round(pct_val, 6),
        "scale": "pct",
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    marker = OUTDIR / f"{INDEX_KEY}_post_intraday.txt"
    if pct_val is None or not math.isfinite(pct_val):
        marker.write_text(f"{INDEX_KEY.upper()} 1d: N/A\n", encoding="utf-8")
    else:
        marker.write_text(f"{INDEX_KEY.upper()} 1d: {pct_val:+.2f}%\n", encoding="utf-8")

# ------------------------
# main
# ------------------------
if __name__ == "__main__":
    gen_pngs_and_stats()
