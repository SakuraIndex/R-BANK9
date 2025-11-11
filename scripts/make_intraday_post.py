#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
R-BANK9 intraday snapshot/post generator (完全版・修正)

入力 CSV:
  docs/outputs/rbank9_intraday.csv
    - ヘッダ: ts,pct
    - 例: 2025-11-11T09:05:00+09:00,0.12

出力:
  - docs/outputs/rbank9_intraday.png
  - docs/outputs/rbank9_post_intraday.txt
  - docs/outputs/rbank9_stats.json

ポイント:
  - ワークフローから古い互換引数（--index-key 等）が来ても受理して無視
  - CSV が空/不正は例外で失敗（公開ステップは止まる）
  - ダークテーマのチャートを出力
"""

from __future__ import annotations
from pathlib import Path
import argparse
import json
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

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # 実際に使う4引数
    p.add_argument("--csv",          default=str(IN_CSV_DEF))
    p.add_argument("--out-json",     dest="out_json",     default=str(OUT_STAT_DEF))
    p.add_argument("--out-text",     dest="out_text",     default=str(OUT_TXT_DEF))
    p.add_argument("--snapshot-png", dest="snapshot_png", default=str(OUT_PNG_DEF))
    # 互換用（無視）
    p.add_argument("--index-key", default=None)
    p.add_argument("--label", default=None)
    p.add_argument("--dt-col", default=None)
    p.add_argument("--value-type", default=None)
    p.add_argument("--basis", default=None)
    p.add_argument("--session-start", default=None)
    p.add_argument("--session-end", default=None)
    p.add_argument("--day-anchor", default=None)
    return p.parse_args()

def setup_dark_theme():
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

def read_csv_strict(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise ValueError(f"CSV がありません: {csv_path}")
    txt = csv_path.read_text(encoding="utf-8").replace("\ufeff", "")
    cleaned = "\n".join(line.strip() for line in txt.splitlines() if line.strip())
    if not cleaned:
        raise ValueError("CSV が空です。")
    csv_path.write_text(cleaned, encoding="utf-8")
    df = pd.read_csv(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}
    if "ts" not in cols or "pct" not in cols:
        raise ValueError(f"CSV ヘッダ不正（ts,pct 必須）: columns={list(df.columns)}")
    df = df.rename(columns={cols["ts"]: "ts", cols["pct"]: "pct"})
    df["ts"]  = pd.to_datetime(df["ts"], errors="coerce", utc=True).dt.tz_convert("Asia/Tokyo")
    df["pct"] = pd.to_numeric(df["pct"], errors="coerce")
    df = df.dropna(subset=["ts", "pct"]).sort_values("ts")
    if df.empty:
        raise ValueError("CSV に有効データがありません。")
    return df

def plot_series(df: pd.DataFrame, out_png: Path):
    setup_dark_theme()
    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    ax.plot(df["ts"], df["pct"])
    ax.set_title(TITLE)
    ax.set_xlabel("Time")
    ax.set_ylabel("Change vs Prev Close (%)")
    for spine in ax.spines.values():
        spine.set_color("#36506b")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def build_post_text(latest_pct: float, label: str) -> str:
    sign = "+" if latest_pct >= 0 else ""
    now_jst = pd.Timestamp.now(tz="Asia/Tokyo").strftime("%Y/%m/%d %H:%M JST")
    return (
        f"▲ {label} 日中スナップショット（{now_jst}）\n"
        f"{sign}{latest_pct:.2f}%（基準: prev_close）\n"
        f"#R_BANK9 #日本株\n"
    )

def build_stats_json(latest_pct: float, index_key: str, label: str) -> str:
    obj = {
        "index_key": index_key,
        "label": label,
        "pct_intraday": latest_pct,
        "basis": "prev_close",
        "session": {"start": "09:00", "end": "15:30", "anchor": "09:00"},
        "updated_at": pd.Timestamp.now(tz="Asia/Tokyo").isoformat(),
    }
    return json.dumps(obj, ensure_ascii=False, indent=2)

def main():
    args = parse_args()

    in_csv   = Path(args.csv) if args.csv else IN_CSV_DEF
    out_png  = Path(args.snapshot_png) if args.snapshot_png else OUT_PNG_DEF  # ← 修正ポイント
    out_txt  = Path(args.out_text) if args.out_text else OUT_TXT_DEF
    out_stat = Path(args.out_json) if args.out_json else OUT_STAT_DEF

    index_key = (args.index_key or INDEX_KEY_DEF)
    label     = (args.label or LABEL_DEF)

    df = read_csv_strict(in_csv)
    latest_pct = float(df["pct"].iloc[-1])

    plot_series(df, out_png)

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text(build_post_text(latest_pct, label), encoding="utf-8")

    out_stat.parent.mkdir(parents=True, exist_ok=True)
    out_stat.write_text(build_stats_json(latest_pct, index_key, label), encoding="utf-8")

    print(f"[ok] rows={len(df)} latest={latest_pct:.4f} -> {out_png}, {out_txt}, {out_stat}")

if __name__ == "__main__":
    main()
