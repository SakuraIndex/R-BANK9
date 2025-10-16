#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R-BANK9 charts + stats  (intraday % from R_BANK9, zero-row guard, dark theme)
"""
from pathlib import Path
import json
from datetime import datetime, timezone
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

# ------------------------
# plotting style (dark)
# ------------------------
DARK_BG = "#0e0f13"
DARK_AX = "#0b0c10"
FG_TEXT = "#e7ecf1"
GRID    = "#2a2e3a"
GREEN   = "#28e07c"
RED     = "#ff4d4d"

def _apply(ax, title: str, ylab: str) -> None:
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
    ax.set_ylabel(ylab, color=FG_TEXT, fontsize=10)

# ------------------------
# data loading helpers
# ------------------------
def _load_df() -> pd.DataFrame:
    """intraday優先でロード。DatetimeIndex化し、列を数値へ強制。"""
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

def _pick_rbank9_col(df: pd.DataFrame) -> str | None:
    """R_BANK9（合成済み）列を最優先で拾う。"""
    candidates = { "rbank9", "r_bank9", "rbnk9", "r-bank9", "R_BANK9", "RBANK9" }
    for c in df.columns:
        if c.strip() in candidates or c.strip().lower() in {x.lower() for x in candidates}:
            return c
    return None

def _first_valid_nonzero(s: pd.Series) -> float | None:
    """先頭から見て NaN/0 をスキップし、最初の正/負の非ゼロ値を返す。"""
    s2 = s.dropna()
    s2 = s2[s2 != 0]
    if len(s2) == 0:
        return None
    return float(s2.iloc[0])

def _intraday_pct_series(df: pd.DataFrame) -> pd.Series | None:
    """
    既存の合成列(R_BANK9)があればそれを%化。
    無ければ、各列の等加重平均(=行ごとの平均)を使う。
    いずれも先頭の 0/NaN を基準から除外して基準値を決める。
    """
    col = _pick_rbank9_col(df)
    if col is not None:
        base_series = df[col]
    else:
        # 等加重平均（その行で非NaNの列のみ）
        base_series = df.mean(axis=1, skipna=True)

    # 先頭のゼロ/NaNを取り除いて基準値決定
    first = _first_valid_nonzero(base_series)
    if first is None:
        return None

    # % 変換（基準未満の先頭側ゼロ/NaNは自然に±∞を避けるため同時に置換）
    pct = (base_series / first - 1.0) * 100.0
    # 可視化の邪魔になる先頭側の NaN は落とす
    pct = pct.dropna()
    return pct

# ------------------------
# chart generation
# ------------------------
def _save_pct(pct: pd.Series, out_png: Path, title: str) -> None:
    # 色は最終値で決める
    last = pct.iloc[-1]
    color = GREEN if last >= 0 else RED

    fig, ax = plt.subplots()
    _apply(ax, title, "Change (%)")
    ax.plot(pct.index, pct.values, color=color, linewidth=1.8)
    fig.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

def gen_pngs() -> float | None:
    df = _load_df()
    pct = _intraday_pct_series(df)
    if pct is None or len(pct) == 0:
        # 画像は空でも良いが、以降の処理用に None を返す
        fig, ax = plt.subplots()
        _apply(ax, f"{INDEX_KEY.upper()} (1d %)", "Change (%)")
        fig.savefig(OUTDIR / f"{INDEX_KEY}_1d.png", bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return None

    # 1d/7d/1m/1y は暫定的に同じスケールで出しておく（必要なら歴史%化も可能）
    _save_pct(pct, OUTDIR / f"{INDEX_KEY}_1d.png", f"{INDEX_KEY.upper()} (1d %)")
    _save_pct(pct, OUTDIR / f"{INDEX_KEY}_7d.png", f"{INDEX_KEY.upper()} (1d %)")
    _save_pct(pct, OUTDIR / f"{INDEX_KEY}_1m.png", f"{INDEX_KEY.upper()} (1d %)")
    _save_pct(pct, OUTDIR / f"{INDEX_KEY}_1y.png", f"{INDEX_KEY.upper()} (1d %)")

    return float(pct.iloc[-1])

# ------------------------
# stats + marker
# ------------------------
def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def write_stats_and_marker(pct_last: float | None) -> None:
    payload = {
        "index_key": INDEX_KEY,
        "pct_1d": None if pct_last is None else round(pct_last, 6),
        "scale": "pct",
        "updated_at": _now_utc_iso(),
    }
    (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )

    # human-readable marker
    marker = OUTDIR / f"{INDEX_KEY}_post_intraday.txt"
    if pct_last is None:
        marker.write_text(f"{INDEX_KEY.upper()} 1d: N/A\n", encoding="utf-8")
    else:
        marker.write_text(f"{INDEX_KEY.upper()} 1d: {pct_last:+.2f}%\n", encoding="utf-8")

# ------------------------
# main
# ------------------------
if __name__ == "__main__":
    last = gen_pngs()
    write_stats_and_marker(last)
