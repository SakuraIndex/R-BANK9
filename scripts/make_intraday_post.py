#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
R-BANK9 intraday snapshot/post generator (robust FINAL)

対応:
- 縦持ち: ts,pct
- 横持ち/壊れやすいCSV:
  - 区切りが , と 全角， 混在
  - 先頭セルが空(NaN)
  - 行内のどこかに時刻セルがあり、その後ろに数値が並ぶ
  - コメントやティッカー/日本語が混在（数値以外は全無視）

出力（指数リポ内; workflow がサイトリポ docs/charts/R_BANK9/ にコピー）:
- docs/outputs/rbank9_intraday.png
- docs/outputs/rbank9_post_intraday.txt
- docs/outputs/rbank9_stats.json
"""

from __future__ import annotations
from pathlib import Path
import argparse
import csv
import json
import re
from typing import List, Optional

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

# 時刻セル検出用（例: 2025-11-11T09:05 / +09:00 はあってもなくてもOK）
TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv",          default=str(IN_CSV_DEF))
    p.add_argument("--out-json",     dest="out_json",     default=str(OUT_STAT_DEF))
    p.add_argument("--out-text",     dest="out_text",     default=str(OUT_TXT_DEF))
    p.add_argument("--snapshot-png", dest="snapshot_png", default=str(OUT_PNG_DEF))
    # 互換用（受理のみ）
    p.add_argument("--index-key", default=None)
    p.add_argument("--label",     default=None)
    p.add_argument("--dt-col",    default=None)
    p.add_argument("--value-type",default=None)
    p.add_argument("--basis",     default=None)
    p.add_argument("--session-start", default=None)
    p.add_argument("--session-end",   default=None)
    p.add_argument("--day-anchor",    default=None)
    return p.parse_args()

# ---------- 基本ユーティリティ ----------
def _clean_num(tok: str) -> Optional[float]:
    if tok is None:
        return None
    s = str(tok).strip().replace("，", ",")  # 全角カンマ->半角
    if not s:
        return None
    # パーセント/カンマ除去
    s = s.replace(",", "").replace("%", "")
    try:
        v = float(s)
        if np.isfinite(v):
            return v
        return None
    except Exception:
        return None

def _to_ts(tok: str) -> Optional[pd.Timestamp]:
    s = (tok or "").strip()
    if not s or not TS_RE.search(s):
        return None
    try:
        ts = pd.to_datetime(s, errors="coerce", utc=True)
        if pd.isna(ts):
            return None
        return ts
    except Exception:
        return None

# ---------- 縦持ち ----------
def read_vertical(path: Path) -> Optional[pd.DataFrame]:
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
    df = df.dropna(subset=["ts","pct"]).sort_values("ts")
    return df if not df.empty else None

# ---------- 横持ち/壊れ耐性 ----------
def read_any_wide(path: Path) -> Optional[pd.DataFrame]:
    rows_ts: List[pd.Timestamp] = []
    rows_pct: List[float] = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        rdr = csv.reader((ln.replace("，", ",") for ln in f))  # 全角区切り→半角に正規化
        lines = list(rdr)

    if len(lines) < 2:
        return None

    # 1 行目はヘッダ（内容は使わない）
    for tokens in lines[1:]:
        if not tokens:
            continue
        # 行内の「最初の時刻セル」を探す
        ts_idx = None
        ts_val = None
        for i, tok in enumerate(tokens):
            ts = _to_ts(tok)
            if ts is not None:
                ts_idx = i
                ts_val = ts
                break
        if ts_idx is None:
            continue

        # 時刻セル以降の数値だけを拾って平均（1 つならそのまま）
        nums: List[float] = []
        for tok in tokens[ts_idx+1:]:
            v = _clean_num(tok)
            if v is not None:
                nums.append(v)
        if not nums:
            continue

        rows_ts.append(ts_val)  # UTC
        rows_pct.append(float(np.mean(nums)))

    if not rows_ts:
        return None

    df = pd.DataFrame({"ts": rows_ts, "pct": rows_pct})
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert("Asia/Tokyo")
    df = df.sort_values("ts")
    return df if not df.empty else None

def read_csv_any(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise ValueError(f"CSV がありません: {path}")
    # 1) 縦持ち
    df = read_vertical(path)
    if df is not None:
        return df
    # 2) 横持ち/壊れ耐性
    df = read_any_wide(path)
    if df is not None:
        return df
    # 3) 失敗時のプレビュー
    try:
        prev = pd.read_csv(path, header=None, nrows=1).iloc[0].tolist()
    except Exception:
        prev = "読み取り失敗"
    raise ValueError(f"CSV 形式を解釈できません。先頭行={prev}")

# ---------- 描画（ダークテーマ） ----------
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
        "grid.alpha":       0.55,
    })

def plot_series(df: pd.DataFrame, out_png: Path):
    _setup_dark()
    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    ax.plot(df["ts"], df["pct"])
    ax.set_title(TITLE)
    ax.set_xlabel("Time")
    ax.set_ylabel("Change vs Prev Close (%)")
    for sp in ax.spines.values():
        sp.set_color("#36506b")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

# ---------- 出力 ----------
def build_post_text(latest: float, label: str) -> str:
    sign = "+" if latest >= 0 else ""
    now_jst = pd.Timestamp.now(tz="Asia/Tokyo").strftime("%Y/%m/%d %H:%M JST")
    return f"▲ {label} 日中スナップショット（{now_jst}）\n{sign}{latest:.2f}%（基準: prev_close）\n#R_BANK9 #日本株\n"

def build_stats_json(latest: float, index_key: str, label: str) -> str:
    obj = {
        "index_key": index_key,
        "label": label,
        "pct_intraday": latest,
        "basis": "prev_close",
        "session": {"start":"09:00","end":"15:30","anchor":"09:00"},
        "updated_at": pd.Timestamp.now(tz="Asia/Tokyo").isoformat(),
    }
    return json.dumps(obj, ensure_ascii=False, indent=2)

# ---------- main ----------
def main():
    a = parse_args()
    in_csv   = Path(a.csv)          if a.csv          else IN_CSV_DEF
    out_png  = Path(a.snapshot_png) if a.snapshot_png else OUT_PNG_DEF
    out_txt  = Path(a.out_text)     if a.out_text     else OUT_TXT_DEF
    out_stat = Path(a.out_json)     if a.out_json     else OUT_STAT_DEF

    index_key = a.index_key or INDEX_KEY_DEF
    label     = a.label     or LABEL_DEF

    df = read_csv_any(in_csv)
    latest = float(df["pct"].iloc[-1])

    plot_series(df, out_png)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text(build_post_text(latest, label), encoding="utf-8")
    out_stat.parent.mkdir(parents=True, exist_ok=True)
    out_stat.write_text(build_stats_json(latest, index_key, label), encoding="utf-8")
    print(f"[ok] rows={len(df)} latest={latest:.4f} -> {out_png}")

if __name__ == "__main__":
    main()
