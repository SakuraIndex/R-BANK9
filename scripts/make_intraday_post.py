#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
R-BANK9 intraday snapshot/post generator

想定CSV:
  docs/outputs/rbank9_intraday.csv
  例）ヘッダ: ts,pct
      2025-11-11T09:05:00+09:00,0.12

出力:
  - *_intraday.png
  - *_post_intraday.txt
  - *_stats.json  （updated_at, pct_intraday を含む）

※ 本スクリプトは CLI 引数で入出力や列名を切替できるようにしてあります。
"""

from __future__ import annotations
from pathlib import Path
import argparse
import json
import math
import sys

import pandas as pd
import matplotlib.pyplot as plt


# ============== ダークテーマ ==============
DARK_BG   = "#0b1420"
DARK_GRID = "#1c2a3a"
FG_TEXT   = "#d4e9f7"
FG_SUB    = "#9fb6c7"

def apply_dark_theme(fig: plt.Figure, ax: plt.Axes):
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    for sp in ax.spines.values():
        sp.set_color(FG_SUB)
    ax.tick_params(colors=FG_TEXT)
    ax.xaxis.label.set_color(FG_TEXT)
    ax.yaxis.label.set_color(FG_TEXT)
    ax.title.set_color(FG_TEXT)


# ============== CSV 読み込み ==============
def read_csv(csv_path: Path, dt_col: str, pct_col: str = "pct") -> pd.DataFrame:
    if not csv_path.exists():
        raise ValueError(f"CSV がありません: {csv_path}")

    raw = csv_path.read_text(encoding="utf-8").replace("\ufeff", "")
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("CSV が空です。")

    csv_path.write_text("\n".join(lines), encoding="utf-8")
    df = pd.read_csv(csv_path)

    # 列名ゆらぎ吸収
    lower = {c.lower().strip(): c for c in df.columns}
    need_dt = dt_col.lower().strip()
    need_pct = pct_col.lower().strip()
    if need_dt not in lower or need_pct not in lower:
        raise ValueError(f"CSV ヘッダに {dt_col},{pct_col} が必要です。columns={list(df.columns)}")

    df = df.rename(columns={lower[need_dt]: "ts", lower[need_pct]: "pct"})
    # 時刻はタイムゾーンを保ったままJSTへ
    ts = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    # すでに +09:00 が付いている文字列は UTC 変換→JST 変換で最終的にJSTへ落ちる
    df["ts"] = ts.dt.tz_convert("Asia/Tokyo")
    df["pct"] = pd.to_numeric(df["pct"], errors="coerce")

    df = df.dropna(subset=["ts", "pct"]).sort_values("ts")
    return df


# ============== プロット ==============
def plot_no_data(out_png: Path, title: str):
    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    apply_dark_theme(fig, ax)
    ax.set_title(title)
    ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, color=FG_TEXT)
    ax.set_xlabel("Time")
    ax.set_ylabel("Change vs Prev Close (%)")
    ax.grid(True, alpha=0.25, color=DARK_GRID)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", facecolor=DARK_BG, edgecolor="none")
    plt.close(fig)


def plot_series(df: pd.DataFrame, out_png: Path, title: str):
    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    apply_dark_theme(fig, ax)

    ax.plot(df["ts"], df["pct"])           # 色はデフォルト任せ
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Change vs Prev Close (%)")
    ax.grid(True, alpha=0.25, color=DARK_GRID)

    # y 軸の見やすい範囲（± 8% に軽くクリップ）
    try:
        ymin = float(df["pct"].min())
        ymax = float(df["pct"].max())
        lo = min(-8.0, math.floor(min(ymin, ymax)))
        hi = max( 8.0, math.ceil (max(ymin, ymax)))
        ax.set_ylim(lo, hi)
    except Exception:
        pass

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", facecolor=DARK_BG, edgecolor="none")
    plt.close(fig)


# ============== 単位補正（auto/ratio/percent） ==============
def normalize_pct(series: pd.Series, value_type: str) -> pd.Series:
    vt = (value_type or "auto").lower()
    s = pd.to_numeric(series, errors="coerce")

    if vt == "percent":
        # そのまま [%] として扱う
        return s
    if vt == "ratio":
        # 比率（0.034 → 3.4%）
        return s * 100.0

    # auto: 値が ±1.2 の範囲なら比率とみなして % 化（安全側）
    return s * 100.0 if s.abs().max() <= 1.2 else s


# ============== メイン ==============
def build_post_text(label: str, basis: str, latest_pct: float) -> str:
    ts_str = pd.Timestamp.now(tz="Asia/Tokyo").strftime("%Y/%m/%d %H:%M")
    sign = "+" if latest_pct >= 0 else ""
    return (
        f"▲ {label} 日中スナップショット（{ts_str} JST）\n"
        f"{sign}{latest_pct:.2f}%（基準: {basis}）\n"
        f"#{label.replace('-', '_')} #日本株\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-key", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--dt-col", default="ts")
    ap.add_argument("--value-type", default="auto")  # auto | ratio | percent
    ap.add_argument("--basis", default="prev_close")

    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-text", required=True)
    ap.add_argument("--snapshot-png", required=True)

    ap.add_argument("--session-start", default="09:00")
    ap.add_argument("--session-end",   default="15:30")
    ap.add_argument("--day-anchor",    default="09:00")
    args = ap.parse_args()

    IN_CSV   = Path(args.csv)
    OUT_PNG  = Path(args.snapshot_png)
    OUT_TXT  = Path(args.out_text)
    OUT_STAT = Path(args.out_json)

    title = f"{args.label} Intraday Snapshot (JST)"

    try:
        df = read_csv(IN_CSV, dt_col=args.dt_col, pct_col="pct")
        if not df.empty:
            # 単位補正
            df["pct"] = normalize_pct(df["pct"], args.value_type)
    except Exception as e:
        # 読めなければ no data
        OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
        plot_no_data(OUT_PNG, title)

        OUT_TXT.write_text(
            f"▲ {args.label} 日中スナップショット（no data）\n"
            f"+0.00%（基準: {args.basis}）\n"
            f"#{args.label.replace('-', '_')} #日本株\n",
            encoding="utf-8",
        )
        OUT_STAT.write_text(json.dumps({
            "index_key": args.index_key,
            "label": args.label,
            "pct_intraday": 0.0,
            "basis": args.basis,
            "session": {"start": args.session_start, "end": args.session_end, "anchor": args.day_anchor},
            "updated_at": pd.Timestamp.now(tz="Asia/Tokyo").isoformat()
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[warn] read_csv failed: {e}", file=sys.stderr)
        return

    latest_pct = float(df["pct"].iloc[-1]) if not df.empty else 0.0

    # 画像
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        plot_no_data(OUT_PNG, title)
    else:
        plot_series(df, OUT_PNG, title)

    # post / stats
    OUT_TXT.write_text(build_post_text(args.label, args.basis, latest_pct), encoding="utf-8")

    OUT_STAT.write_text(json.dumps({
        "index_key": args.index_key,
        "label": args.label,
        "pct_intraday": latest_pct,
        "basis": args.basis,
        "session": {"start": args.session_start, "end": args.session_end, "anchor": args.day_anchor},
        "updated_at": pd.Timestamp.now(tz="Asia/Tokyo").isoformat()
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
