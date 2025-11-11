#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
R-BANK9 intraday snapshot/post generator (ultimate robust)

入力CSVが壊れ/混在していても行ベースで自力パース:
- パターンA: 縦持ち(ts,pct)
- パターンB: 行内のどこかに時刻(YYYY-MM-DDTHH:MM)があり、以降に数値が並ぶ
- 壊れ行/銘柄行/コメントは全スキップ

出力（指数リポ内; workflow がサイトリポ docs/charts/R_BANK9/ へコピー）:
- docs/outputs/rbank9_intraday.png
- docs/outputs/rbank9_post_intraday.txt
- docs/outputs/rbank9_stats.json
"""

from __future__ import annotations
from pathlib import Path
import argparse
import json
import re
from typing import List, Optional

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

TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}")          # 2025-11-11T09:05
HDR_RE = re.compile(r"^\s*ts\s*,\s*pct\s*$", re.IGNORECASE)  # ts,pct だけの行

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv",          default=str(IN_CSV_DEF))
    p.add_argument("--out-json",     dest="out_json",     default=str(OUT_STAT_DEF))
    p.add_argument("--out-text",     dest="out_text",     default=str(OUT_TXT_DEF))
    p.add_argument("--snapshot-png", dest="snapshot_png", default=str(OUT_PNG_DEF))
    # 互換で受理のみ（値は使わない）
    p.add_argument("--index-key", default=None)
    p.add_argument("--label",     default=None)
    p.add_argument("--dt-col",    default=None)
    p.add_argument("--value-type",default=None)
    p.add_argument("--basis",     default=None)
    p.add_argument("--session-start", default=None)
    p.add_argument("--session-end",   default=None)
    p.add_argument("--day-anchor",    default=None)
    return p.parse_args()

# ---------------- line parser ----------------

def _clean_num(tok: str) -> Optional[float]:
    if tok is None:
        return None
    s = str(tok).strip().replace("，", ",")  # 全角 , → 半角
    if not s:
        return None
    s = s.replace(",", "").replace("%", "")
    try:
        v = float(s)
        return float(v) if np.isfinite(v) else None
    except Exception:
        return None

def _parse_vertical(lines: List[str], start_idx: int) -> List[tuple]:
    """'ts,pct' ヘッダ行の次行から末尾までを ts,pct として取り込む"""
    rows: List[tuple] = []
    for ln in lines[start_idx+1:]:
        ln = ln.strip()
        if not ln:
            continue
        ln = ln.replace("，", ",")
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 2:
            continue
        ts_raw, pct_raw = parts[0], parts[1]
        if not TS_RE.search(ts_raw):
            continue
        try:
            ts = pd.to_datetime(ts_raw, errors="coerce", utc=True)
            if pd.isna(ts):
                continue
            pct = _clean_num(pct_raw)
            if pct is None:
                continue
            rows.append((ts, pct))
        except Exception:
            continue
    return rows

def _parse_any(lines: List[str]) -> List[tuple]:
    """縦持ち or 横持ち(行内時刻 + 数値群) を両対応で吸い上げ"""
    rows: List[tuple] = []

    # まず ts,pct ヘッダを探す
    for i, ln in enumerate(lines):
        if HDR_RE.match(ln.strip().replace("，", ",")):
            rows_v = _parse_vertical(lines, i)
            if rows_v:
                rows.extend(rows_v)
            break  # ヘッダは1箇所想定。見つけたら縦持ち優先しつつ、後段の横持ちも続行

    # 全行を横持ちルールで走査
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        s = s.replace("，", ",")
        # ts を検出
        m = TS_RE.search(s)
        if not m:
            continue
        ts_raw = m.group(0)
        try:
            ts = pd.to_datetime(ts_raw, errors="coerce", utc=True)
            if pd.isna(ts):
                continue
        except Exception:
            continue

        # ts 以降のカンマ区切りから数値だけ抽出
        tail = s[m.end():]
        toks = [t.strip() for t in tail.split(",") if t.strip()]
        nums = [v for v in (_clean_num(t) for t in toks) if v is not None]
        if not nums:
            # 縦持ちの ts,pct 行だった場合、すでに縦持ちで吸っていればスキップ
            continue

        rows.append((ts, float(np.mean(nums))))

    return rows

def read_csv_super_robust(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise ValueError(f"CSV がありません: {path}")
    text = path.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        raise ValueError("CSV が空です。")

    # 行列作成
    lines = [ln.rstrip("\n") for ln in text.splitlines() if ln.strip()]
    rows = _parse_any(lines)

    if not rows:
        # デバッグ用プレビュー（最初の行だけ）
        prev = lines[0] if lines else "(empty)"
        raise ValueError(f"CSV 形式を解釈できません。先頭行={prev}")

    ts_list = [r[0] for r in rows]
    pct_list = [r[1] for r in rows]
    df = pd.DataFrame({"ts": ts_list, "pct": pct_list})
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert("Asia/Tokyo")
    df = df.sort_values("ts")
    return df

# ---------------- plotting (dark theme) ----------------

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
    if df.empty:
        ax.set_title(TITLE)
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Time")
        ax.set_ylabel("Change vs Prev Close (%)")
        for sp in ax.spines.values():
            sp.set_color("#36506b")
        fig.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        return

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

# ---------------- outputs ----------------

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
    in_csv   = Path(a.csv)          if a.csv          else IN_CSV_DEF
    out_png  = Path(a.snapshot_png) if a.snapshot_png else OUT_PNG_DEF
    out_txt  = Path(a.out_text)     if a.out_text     else OUT_TXT_DEF
    out_stat = Path(a.out_json)     if a.out_json     else OUT_STAT_DEF

    index_key = a.index_key or INDEX_KEY_DEF
    label     = a.label     or LABEL_DEF

    try:
        df = read_csv_super_robust(in_csv)
    except Exception as e:
        # 失敗時でも「no data」で出力（ダークテーマ）
        plot_series(pd.DataFrame(columns=["ts","pct"]), out_png)
        out_txt.parent.mkdir(parents=True, exist_ok=True)
        out_txt.write_text(f"▲ {label} 日中スナップショット (no data)\n+0.00%（基準: prev_close）\n#R_BANK9 #日本株\n", encoding="utf-8")
        out_stat.parent.mkdir(parents=True, exist_ok=True)
        out_stat.write_text(build_stats_json(0.0, index_key, label), encoding="utf-8")
        print(f"[warn] parse failed: {e}")
        return

    latest = float(df["pct"].iloc[-1]) if not df.empty else 0.0
    plot_series(df, out_png)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text(build_post_text(latest, label), encoding="utf-8")
    out_stat.parent.mkdir(parents=True, exist_ok=True)
    out_stat.write_text(build_stats_json(latest, index_key, label), encoding="utf-8")
    print(f"[ok] rows={len(df)} latest={latest:.4f} -> {out_png}")

if __name__ == "__main__":
    main()
