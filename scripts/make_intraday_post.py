#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
R-BANK9 intraday snapshot/post generator (最終版)

受け入れ形式:
  1) 縦持ち: ヘッダ ts,pct
  2) 横持ち: 先頭行 = [, 5830.T, # コメント, 5831.T, # コメント, ...]
             2行目以降 = "ISO時刻, 各ティッカーの%値 …"
    → 先頭セルが空でもOK。#列は自動的に無視。NaNは除外して等加重平均を算出。

出力:
  docs/outputs/rbank9_intraday.png
  docs/outputs/rbank9_post_intraday.txt
  docs/outputs/rbank9_stats.json
"""

from __future__ import annotations
from pathlib import Path
import argparse
import json
import re
import math
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 既定パス
IN_CSV_DEF   = Path("docs/outputs/rbank9_intraday.csv")
OUT_PNG_DEF  = Path("docs/outputs/rbank9_intraday.png")
OUT_TXT_DEF  = Path("docs/outputs/rbank9_post_intraday.txt")
OUT_STAT_DEF = Path("docs/outputs/rbank9_stats.json")

INDEX_KEY_DEF = "R_BANK9"
LABEL_DEF     = "R-BANK9"
TITLE         = "R-BANK9 Intraday Snapshot (JST)"

# ---------- 引数 ----------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv",          default=str(IN_CSV_DEF))
    p.add_argument("--out-json",     dest="out_json",     default=str(OUT_STAT_DEF))
    p.add_argument("--out-text",     dest="out_text",     default=str(OUT_TXT_DEF))
    p.add_argument("--snapshot-png", dest="snapshot_png", default=str(OUT_PNG_DEF))
    # 互換用（値は使わない）
    p.add_argument("--index-key", default=None)
    p.add_argument("--label", default=None)
    p.add_argument("--dt-col", default=None)
    p.add_argument("--value-type", default=None)
    p.add_argument("--basis", default=None)
    p.add_argument("--session-start", default=None)
    p.add_argument("--session-end", default=None)
    p.add_argument("--day-anchor", default=None)
    return p.parse_args()

# ---------- ユーティリティ ----------
_TICKER_RE = re.compile(r"\.T\s*$", re.IGNORECASE)

def _clean_number(s: str) -> float | None:
    """ ' 0.12 ' / '0,12' / '0.12%' / '' → float or None """
    if s is None:
        return None
    x = str(s).strip()
    if not x or x.lower() == "nan":
        return None
    x = x.replace(",", "").replace("%", "")
    try:
        return float(x)
    except Exception:
        return None

def _lines_of(path: Path) -> List[str]:
    txt = path.read_text(encoding="utf-8", errors="ignore").replace("\ufeff", "")
    # 完全空行は捨てる・前後空白は温存（先頭カンマを壊さない）
    return [ln.rstrip("\n\r") for ln in txt.splitlines() if ln.strip()]

# ---------- 縦持ち(ts,pct)を読む ----------
def _read_vertical(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    cols = {c.lower().strip(): c for c in df.columns}
    if "ts" not in cols or "pct" not in cols:
        return None
    df = df.rename(columns={cols["ts"]: "ts", cols["pct"]: "pct"})
    df["ts"]  = pd.to_datetime(df["ts"], errors="coerce", utc=True).dt.tz_convert("Asia/Tokyo")
    df["pct"] = pd.to_numeric(df["pct"], errors="coerce")
    df = df.dropna(subset=["ts", "pct"]).sort_values("ts")
    return df if not df.empty else None

# ---------- 横持ちを手書きでパース ----------
def _read_wide_text(path: Path) -> pd.DataFrame | None:
    lines = _lines_of(path)
    if len(lines) < 2:
        return None

    # 先頭行をカンマで素朴に分割（引用符は想定しない）
    header_cells = [c.strip() for c in lines[0].split(",")]
    # 先頭セルが空でもOK。2セル目以降で .T or '# ' を探す
    has_ticker_or_hash = any(_TICKER_RE.search(c) or c.startswith("#") for c in header_cells[1:])
    if not has_ticker_or_hash:
        return None

    # ティッカー列のインデックス（1 origin 側にある）
    ticker_idx: List[int] = [i for i, c in enumerate(header_cells)
                             if _TICKER_RE.search(c)]
    if not ticker_idx:
        return None

    # データ行を回して ts と 各ティッカー列の値 を拾い、等加重平均
    ts_list: List[pd.Timestamp] = []
    pct_list: List[float] = []

    for ln in lines[1:]:
        cells = [c.strip() for c in ln.split(",")]
        if not cells:
            continue
        ts_s = cells[0] if len(cells) > 0 else ""
        try:
            ts = pd.to_datetime(ts_s, errors="coerce", utc=True)
        except Exception:
            ts = pd.NaT
        if pd.isna(ts):
            continue

        vals: List[float] = []
        for idx in ticker_idx:
            if idx >= len(cells):
                continue
            v = _clean_number(cells[idx])
            if v is None or math.isnan(v):
                continue
            vals.append(v)

        if not vals:
            # 値が全欠損ならスキップ
            continue

        ts_list.append(ts)
        pct_list.append(float(np.mean(vals)))

    if not ts_list:
        return None

    df = pd.DataFrame({"ts": ts_list, "pct": pct_list})
    # UTC → JST
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert("Asia/Tokyo")
    df = df.sort_values("ts")
    return df if not df.empty else None

# ---------- 総合入口 ----------
def read_csv_any(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise ValueError(f"CSV がありません: {path}")

    # 1) まず縦持ち
    df = _read_vertical(path)
    if df is not None:
        return df

    # 2) 横持ち（手書きパーサ）
    df = _read_wide_text(path)
    if df is not None:
        return df

    # 3) デバッグ用に先頭行を見せる
    try:
        preview = pd.read_csv(path, nrows=1, header=None).iloc[0].tolist()
    except Exception:
        preview = "読み取り失敗"
    raise ValueError(f"CSV 形式を解釈できません。先頭行={preview}")

# ---------- 描画 ----------
def _setup_dark():
    plt.rcParams.update({
        "figure.facecolor": "#0b1420",
        "axes.facecolor":   "#0b1420",
        "savefig.facecolor":"#0b1420",
        "text.color":       "#d4e9f7",
        "axes.labelcolor":  "#d4e9f7",
        "axes.edgecolor":   "#6b7f91",
        "xtick.color":      "#c8d7e2",
        "ytick.color":      "#c8d7e2",
        "grid.color":       "#274057",
        "axes.grid":        True,
        "grid.alpha":       0.6,
    })

def plot_series(df: pd.DataFrame, out_png: Path):
    _setup_dark()
    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    ax.plot(df["ts"], df["pct"])
    ax.set_title("R-BANK9 Intraday Snapshot (JST)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Change vs Prev Close (%)")
    for sp in ax.spines.values():
        sp.set_color("#36506b")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

# ---------- 出力 ----------
def _post_text(latest: float, label: str) -> str:
    sign = "+" if latest >= 0 else ""
    now_jst = pd.Timestamp.now(tz="Asia/Tokyo").strftime("%Y/%m/%d %H:%M JST")
    return f"▲ {label} 日中スナップショット（{now_jst}）\n{sign}{latest:.2f}%（基準: prev_close）\n#R_BANK9 #日本株\n"

def _stats_json(latest: float, index_key: str, label: str) -> str:
    obj = {
        "index_key": index_key,
        "label": label,
        "pct_intraday": latest,
        "basis": "prev_close",
        "session": {"start": "09:00", "end": "15:30", "anchor": "09:00"},
        "updated_at": pd.Timestamp.now(tz="Asia/Tokyo").isoformat(),
    }
    return json.dumps(obj, ensure_ascii=False, indent=2)

# ---------- main ----------
def main():
    a = parse_args()
    in_csv   = Path(a.csv) if a.csv else IN_CSV_DEF
    out_png  = Path(a.snapshot_png) if a.snapshot_png else OUT_PNG_DEF
    out_txt  = Path(a.out_text) if a.out_text else OUT_TXT_DEF
    out_stat = Path(a.out_json) if a.out_json else OUT_STAT_DEF

    index_key = (a.index_key or INDEX_KEY_DEF)
    label     = (a.label or LABEL_DEF)

    df = read_csv_any(in_csv)
    latest = float(df["pct"].iloc[-1])

    plot_series(df, out_png)

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text(_post_text(latest, label), encoding="utf-8")

    out_stat.parent.mkdir(parents=True, exist_ok=True)
    out_stat.write_text(_stats_json(latest, index_key, label), encoding="utf-8")

    print(f"[ok] rows={len(df)}, latest={latest:.4f}, out={out_png}")

if __name__ == "__main__":
    main()
