# scripts/long_charts.py
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import timezone
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# ====== 設定 ======
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_OUT = PROJECT_ROOT / "docs" / "outputs"

INDEX_KEY = "rbank9"         # 出力ファイル名の接頭辞
CSV_INTRADAY = DOCS_OUT / f"{INDEX_KEY}_intraday.csv"
PNG_1D = DOCS_OUT / f"{INDEX_KEY}_1d.png"
TXT_POST = DOCS_OUT / f"{INDEX_KEY}_post_intraday.txt"
JSON_STATS = DOCS_OUT / f"{INDEX_KEY}_stats.json"

MARKET_TZ = "Asia/Tokyo"     # Chartの目盛り表示用
LEVEL_COL = "R_BANK9"        # CSV のインデックス列名（最後の列）

# ====== ユーティリティ ======
@dataclass
class PctResult:
    pct: Optional[float]          # 1d %（例: +1.07 -> 1.07）。基準なしは None
    basis_note: str               # 基準の説明（例: "basis 00:05 used first valid"）
    first_ts: Optional[pd.Timestamp]
    last_ts: Optional[pd.Timestamp]

def _read_intraday(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # 先頭列が時刻
    ts_col = df.columns[0]
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.set_index(ts_col).sort_index()
    # 数値以外を落としておく
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _to_market_tz(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    # 既に tz-aware 想定（UTC）。マーケットのタイムゾーンへ表示用に convert
    return idx.tz_convert(MARKET_TZ)

def _first_last_valid(s: pd.Series) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    valid = s.replace([math.inf, -math.inf], pd.NA).dropna()
    if valid.empty:
        return None, None
    return valid.index[0], valid.index[-1]

def _calc_1d_pct_from_level(s: pd.Series) -> PctResult:
    """先頭の有効な level を基準に固定して、最後の有効値との比で 1d % を出す。"""
    # 有効データのみ（>0 & finite）
    s2 = s.replace([math.inf, -math.inf], pd.NA).dropna()
    s2 = s2[s2 > 0]
    if s2.empty:
        return PctResult(None, "basis invalid=N/A", None, None)

    first_ts, last_ts = s2.index[0], s2.index[-1]
    first_v, last_v = float(s2.iloc[0]), float(s2.iloc[-1])

    if first_v <= 0 or not math.isfinite(first_v) or not math.isfinite(last_v):
        return PctResult(None, "basis invalid=N/A", first_ts, last_ts)

    pct = (last_v / first_v - 1.0) * 100.0
    basis_note = "basis first-row invalid=used first valid" if s.index[0] != first_ts else "basis first row"
    return PctResult(pct, basis_note, first_ts, last_ts)

def _line_color(pct: Optional[float]) -> str:
    if pct is None:
        return "#22c55e"  # デフォは緑
    return "#22c55e" if pct >= 0 else "#ef4444"

# ====== 描画 ======
def plot_level(df: pd.DataFrame, level_col: str, pct_res: PctResult, out_png: Path) -> None:
    s = df[level_col].copy()
    idx_local = _to_market_tz(s.index)

    plt.figure(figsize=(12, 6), dpi=150)
    plt.plot(idx_local, s, color=_line_color(pct_res.pct), linewidth=2)
    plt.title(f"{INDEX_KEY.upper()} (1d level)")
    plt.xlabel("Time")
    plt.ylabel("Index (level)")
    plt.grid(True, alpha=0.25)

    # 基準時刻を縦線で補助表示（分かりやすさのため・データがある場合のみ）
    if pct_res.first_ts is not None:
        first_local = _to_market_tz(pd.DatetimeIndex([pct_res.first_ts]))[0]
        plt.axvline(first_local, linestyle="--", alpha=0.3)

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()

# ====== 出力 ======
def write_post_txt(pct_res: PctResult, out_txt: Path) -> None:
    if pct_res.pct is None:
        line = f"{INDEX_KEY.upper()} 1d: N/A ({pct_res.basis_note})"
    else:
        sign = "+" if pct_res.pct >= 0 else ""
        line = f"{INDEX_KEY.upper()} 1d: {sign}{pct_res.pct:.2f}% ({pct_res.basis_note})"
    out_txt.write_text(line + "\n", encoding="utf-8")

def write_stats_json(pct_res: PctResult, out_json: Path) -> None:
    payload = {
        "index_key": INDEX_KEY,
        "pct_1d": None if pct_res.pct is None or not math.isfinite(pct_res.pct) else round(pct_res.pct, 6),
        "scale": "level",  # これでダッシュボード側も「レベル指標」を認識できます
        "updated_at": pd.Timestamp.utcnow().isoformat(timespec="seconds") + "Z",
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

# ====== メイン ======
def main() -> None:
    if not CSV_INTRADAY.exists():
        raise FileNotFoundError(f"missing: {CSV_INTRADAY}")

    df = _read_intraday(CSV_INTRADAY)
    if LEVEL_COL not in df.columns:
        raise ValueError(f"CSVに '{LEVEL_COL}' 列が見つかりません")

    # 1d 騰落率（基準＝最初の有効値）
    pct_res = _calc_1d_pct_from_level(df[LEVEL_COL])

    # チャート（y軸はそのまま level）
    plot_level(df, LEVEL_COL, pct_res, PNG_1D)

    # テキスト／JSON（どちらも基準注記付き）
    write_post_txt(pct_res, TXT_POST)
    write_stats_json(pct_res, JSON_STATS)

if __name__ == "__main__":
    main()
