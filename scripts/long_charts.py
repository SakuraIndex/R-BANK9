#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R-BANK9 charts + stats
 - 1d は Change(%) を描画（サイトの騰落率と一致）
 - 7d/1m/1y はレベルを描画（従来どおり）
 - マーケット日の判定は「現在」ではなく CSV の最新時刻ベース
 - ダークテーマ／自動ライン色（上昇=緑、下落=赤）
"""
from pathlib import Path
import json
from datetime import datetime, timezone, timedelta
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------
# constants / paths
# ------------------------
INDEX_KEY = "rbank9"
MARKET_TZ = "Asia/Tokyo"  # 東京市場
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

def _save_line(x, y, out_png: Path, title: str, y_label: str, color=None) -> None:
    fig, ax = plt.subplots()
    _apply(ax, title, y_label)
    if len(x) > 0:
        if color is None:
            color = FG_TEXT
        ax.plot(x, y, color=color, linewidth=1.6)
    fig.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

# ------------------------
# data loading helpers
# ------------------------
def _pick_index_column(df: pd.DataFrame) -> str:
    """R-BANK9 列名を推定。無ければ最後の列。"""
    cand = {
        "rbank9", "r_bank9", "rbnk9", "rbank_9", "r_bank_9", "r-bank9",
        INDEX_KEY, INDEX_KEY.upper(), "R_BANK9", "RBANK9"
    }
    for c in df.columns:
        if c and c.strip().lower() in cand:
            return c
    return df.columns[-1]

def _load_df() -> pd.DataFrame:
    """
    intraday があれば intraday 優先、無ければ history。
    先頭列を DatetimeIndex（tz-aware）にし、数値列へ変換。
    """
    if INTRADAY_CSV.exists():
        df = pd.read_csv(INTRADAY_CSV, parse_dates=[0], index_col=0)
    elif HISTORY_CSV.exists():
        df = pd.read_csv(HISTORY_CSV, parse_dates=[0], index_col=0)
    else:
        raise FileNotFoundError("R-BANK9: neither intraday nor history csv found.")

    # tz を必ず持たせる（無ければ UTC とみなす）
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    # 数値化
    for c in list(df.columns):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(how="all")
    return df

# ------------------------
# core logic
# ------------------------
def _latest_market_day_slice(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    CSV の最新行の「市場タイムゾーンでの日付」を基準に、その日のデータだけ切り出す。
    これにより、JSTの翌日に手動実行しても “前日データ” を正しく 1d と認識できる。
    """
    if len(df.index) == 0:
        return df

    idx_mkt = df.index.tz_convert(MARKET_TZ)
    latest_local = idx_mkt.max()
    day_start = latest_local.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + timedelta(days=1)

    # マスクは market tz ベースで評価
    mask = (idx_mkt >= day_start) & (idx_mkt < day_end)
    day = df.loc[mask]
    return day

def _first_valid_positive(series: pd.Series):
    """0/NaN を除いて最初の正の値を返す（無ければ None）。"""
    s = series.dropna()
    s = s[s > 0]
    return None if s.empty else float(s.iloc[0])

def gen_pngs_and_stats() -> None:
    df = _load_df()
    if len(df.index) == 0:
        # 何も描けないが空で保存
        _save_line([], [], OUTDIR / f"{INDEX_KEY}_1d.png", f"{INDEX_KEY.upper()} (1d %)", "Change (%)")
        _save_line([], [], OUTDIR / f"{INDEX_KEY}_7d.png", f"{INDEX_KEY.upper()} (7d)", "Index / Value")
        _save_line([], [], OUTDIR / f"{INDEX_KEY}_1m.png", f"{INDEX_KEY.upper()} (1m)", "Index / Value")
        _save_line([], [], OUTDIR / f"{INDEX_KEY}_1y.png", f"{INDEX_KEY.upper()} (1y)", "Index / Value")
        (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(
            json.dumps({"index_key": INDEX_KEY, "pct_1d": None, "scale": "pct", "updated_at": _now_utc_iso()}, ensure_ascii=False),
            encoding="utf-8",
        )
        (OUTDIR / f"{INDEX_KEY}_post_intraday.txt").write_text(f"{INDEX_KEY.upper()} 1d: N/A\n", encoding="utf-8")
        return

    col = _pick_index_column(df)

    # 1) 1d: % シリーズ生成（最新ローカル日）
    day = _latest_market_day_slice(df, col)
    pct_series = pd.Series(dtype=float)
    pct_last = None

    if len(day.index) >= 2:
        first = _first_valid_positive(day[col])
        if first is not None:
            pct_series = (day[col] / first - 1.0) * 100.0
            pct_series = pct_series.dropna()
            if not pct_series.empty:
                pct_last = float(pct_series.iloc[-1])

    # 1d の色と保存
    if not pct_series.empty:
        color_1d = GREEN if pct_series.iloc[-1] >= 0 else RED
        _save_line(pct_series.index, pct_series.values,
                   OUTDIR / f"{INDEX_KEY}_1d.png",
                   f"{INDEX_KEY.upper()} (1d %)", "Change (%)", color=color_1d)
    else:
        # 空でも枠は保存
        _save_line([], [], OUTDIR / f"{INDEX_KEY}_1d.png", f"{INDEX_KEY.upper()} (1d %)", "Change (%)")

    # 2) 7d/1m/1y: レベル描画（従来どおり）
    def _save_level_window(nrows: int, outfile: str, title: str):
        win = df.tail(nrows)
        if len(win.index) >= 2:
            line_color = GREEN if win[col].iloc[-1] >= win[col].iloc[0] else RED
            _save_line(win.index, win[col].values, OUTDIR / outfile, title, "Index / Value", color=line_color)
        else:
            _save_line([], [], OUTDIR / outfile, title, "Index / Value")

    _save_level_window(7 * 1000,  f"{INDEX_KEY}_7d.png", f"{INDEX_KEY.upper()} (7d)")
    _save_level_window(30 * 1000, f"{INDEX_KEY}_1m.png", f"{INDEX_KEY.upper()} (1m)")
    _save_level_window(365 * 1000,f"{INDEX_KEY}_1y.png", f"{INDEX_KEY.upper()} (1y)")

    # 3) stats / marker
    payload = {
        "index_key": INDEX_KEY,
        "pct_1d": None if pct_last is None else round(pct_last, 6),
        "scale": "pct",
        "updated_at": _now_utc_iso(),
    }
    (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    if pct_last is None:
        (OUTDIR / f"{INDEX_KEY}_post_intraday.txt").write_text(f"{INDEX_KEY.upper()} 1d: N/A\n", encoding="utf-8")
    else:
        (OUTDIR / f"{INDEX_KEY}_post_intraday.txt").write_text(f"{INDEX_KEY.upper()} 1d: {pct_last:+.2f}%\n", encoding="utf-8")

# ------------------------
# utils
# ------------------------
def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

# ------------------------
# main
# ------------------------
if __name__ == "__main__":
    gen_pngs_and_stats()
