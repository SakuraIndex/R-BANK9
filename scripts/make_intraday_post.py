#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
R-BANK9 intraday snapshot/post generator (robust final)

対応フォーマット
- 縦持ち:   ts,pct
- 横持ち:   先頭行にティッカー/コメント、2 行目以降は
            「時刻, 値, 値, …, #コメント, …」
  ※ 2 列目がすでに集計値でも OK（2 列目以降の数値セルを平均。1 つならそのまま）

出力
- docs/outputs/rbank9_intraday.png（ダークテーマ）
- docs/outputs/rbank9_post_intraday.txt
- docs/outputs/rbank9_stats.json
"""

from __future__ import annotations
from pathlib import Path
import argparse
import json
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

IN_CSV_DEF   = Path("docs/outputs/rbank9_intraday.csv")
OUT_PNG_DEF  = Path("docs/outputs/rbank9_intraday.png")
OUT_TXT_DEF  = Path("docs/outputs/rbank9_post_intraday.txt")
OUT_STAT_DEF = Path("docs/outputs/rbank9_stats.json")

INDEX_KEY_DEF = "R_BANK9"
LABEL_DEF     = "R-BANK9"
TITLE         = "R-BANK9 Intraday Snapshot (JST)"

# ---------------- args ----------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv",          default=str(IN_CSV_DEF))
    p.add_argument("--out-json",     dest="out_json",     default=str(OUT_STAT_DEF))
    p.add_argument("--out-text",     dest="out_text",     default=str(OUT_TXT_DEF))
    p.add_argument("--snapshot-png", dest="snapshot_png", default=str(OUT_PNG_DEF))
    # 互換用（値は使わない・受け取るだけ）
    p.add_argument("--index-key", default=None)
    p.add_argument("--label", default=None)
    p.add_argument("--dt-col", default=None)
    p.add_argument("--value-type", default=None)
    p.add_argument("--basis", default=None)
    p.add_argument("--session-start", default=None)
    p.add_argument("--session-end", default=None)
    p.add_argument("--day-anchor", default=None)
    return p.parse_args()

# ---------------- helpers ----------------
def _clean_num(x: str) -> float | None:
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return None
    s = s.replace(",", "").replace("%", "")
    try:
        return float(s)
    except Exception:
        return None

def _read_text_lines(p: Path) -> List[str]:
    txt = p.read_text(encoding="utf-8", errors="ignore").replace("\ufeff", "")
    # 空行は除去、先頭のカンマ（空セル）は壊さないよう strip はしない
    return [ln.rstrip("\r\n") for ln in txt.splitlines() if ln.strip()]

# ---------------- readers ----------------
def read_vertical(path: Path) -> pd.DataFrame | None:
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

def read_text_generic(path: Path) -> pd.DataFrame | None:
    """先頭列=時刻、2 列目以降=“数値セルだけ”を拾って平均（1 つならそのまま）。"""
    lines = _read_text_lines(path)
    if len(lines) < 2:
        return None

    ts_list: List[pd.Timestamp] = []
    pct_list: List[float] = []

    # 1 行目はヘッダ（内容は使わない）
    for ln in lines[1:]:
        cells = [c.strip() for c in ln.split(",")]
        if not cells:
            continue
        ts_s = cells[0] if len(cells) > 0 else ""
        ts = pd.to_datetime(ts_s, errors="coerce", utc=True)
        if pd.isna(ts):
            continue

        nums = []
        for c in cells[1:]:
            v = _clean_num(c)
            if v is not None and np.isfinite(v):
                nums.append(float(v))

        if not nums:
            # 数値が 1 つもない行は無視
            continue

        ts_list.append(ts)
        pct_list.append(float(np.mean(nums)))  # 1 つならその値、複数なら平均

    if not ts_list:
        return None

    df = pd.DataFrame({"ts": ts_list, "pct": pct_list})
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert("Asia/Tokyo")
    df = df.sort_values("ts")
    return df if not df.empty else None

def read_csv_any(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise ValueError(f"CSV がありません: {path}")

    # 1) 縦持ち優先
    df = read_vertical(path)
    if df is not None:
        return df

    # 2) 横持ち/ゆるい形式
    df = read_text_generic(path)
    if df is not None:
        return df

    # 3) どうしても読めない時はプレビューを埋め込んで例外
    try:
        preview = pd.read_csv(path, nrows=1, header=None).iloc[0].tolist()
    except Exception:
        preview = "読み取り失敗"
    raise ValueError(f"CSV 形式を解釈できません。先頭行={preview}")

# ---------------- plot ----------------
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
    ax.set_title(TITLE)
    ax.set_xlabel("Time")
    ax.set_label = "Change vs Prev Close (%)"
    ax.set_ylabel("Change vs Prev Close (%)")
    for sp in ax.spines.values():
        sp.set_color("#36506b")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

# ---------------- output ----------------
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
        "session": {"start": "09:00", "end": "15:30", "anchor": "09:00"},
        "updated_at": pd.Timestamp.now(tz="Asia/Tokyo").isoformat(),
    }
    return json.dumps(obj, ensure_ascii=False, indent=2)

# ---------------- main ----------------
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
    out_txt.write_text(build_post_text(latest, label), encoding="utf-8")

    out_stat.parent.mkdir(parents=True, exist_ok=True)
    out_stat.write_text(build_stats_json(latest, index_key, label), encoding="utf-8")

    print(f"[ok] rows={len(df)} latest={latest:.4f} file={out_png}")

if __name__ == "__main__":
    main()
