#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
R-BANK9 intraday snapshot/post generator (robust)
- 入力 CSV は rbank9_intraday.csv （推奨: ヘッダ ts,pct / 値は % ）
- コメント行(#...)・ゴミ行・空行を無視して読み取る
- 出力:
  * site_intraday.png
  * site_post.txt
  * site_stats.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import io
import pandas as pd
import matplotlib.pyplot as plt

TZ = "Asia/Tokyo"

# ====== ダークテーマ（枠線なし） ======
BG = "#0b1220"      # 背景
FG = "#d1d5db"      # 文字色
GRID = "#334155"    # グリッド

def set_dark_axes(ax):
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=FG, labelcolor=FG)
    ax.grid(True, alpha=0.35, color=GRID, linestyle="-", linewidth=0.7)
    ax.title.set_color(FG)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)

def read_ts_pct(csv_path: Path) -> pd.DataFrame:
    """
    ts,pct を取り出して昇順に整える。
    - コメント行(#...)や空行を除去
    - 1 行にカンマが 1 個以外なら捨てる（= ノイズを除外）
    - ts は tz-aware に、pct は float に
    """
    if not csv_path.exists():
        return pd.DataFrame(columns=["ts", "pct"])

    raw = csv_path.read_text(encoding="utf-8", errors="ignore")
    # ゴミ行を落とし込み
    cleaned_lines = []
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            continue
        # ヘッダは ts,pct のみ許可（それ以外の文字列ヘッダはスキップ）
        if ("," not in s) or (s.count(",") != 1):
            continue
        cleaned_lines.append(s)

    if not cleaned_lines:
        return pd.DataFrame(columns=["ts", "pct"])

    buf = io.StringIO("\n".join(cleaned_lines))
    try:
        df = pd.read_csv(buf, header=None, names=["ts", "pct"])
    except Exception:
        return pd.DataFrame(columns=["ts", "pct"])

    # 文字列→型
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    # tz を Asia/Tokyo に変換（元が +09:00 ならそのまま扱える）
    df["ts"] = df["ts"].dt.tz_convert(TZ)
    df["pct"] = pd.to_numeric(df["pct"], errors="coerce")

    df = df.dropna(subset=["ts", "pct"]).sort_values("ts").reset_index(drop=True)
    return df

def latest_pct(df: pd.DataFrame) -> float | None:
    if df is None or df.empty:
        return None
    return float(df["pct"].iloc[-1])

def render_chart(df: pd.DataFrame, out_png: Path, label: str):
    # 図の背景もダークに
    plt.rcParams["figure.facecolor"] = BG

    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    set_dark_axes(ax)

    title = f"{label} Intraday Snapshot (JST)"
    if df.empty:
        ax.set_title(title + " (no data)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Change vs Prev Close (%)")
        # 何も描かない（白い枠線が出ない）
    else:
        ax.plot(df["ts"], df["pct"], linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Change vs Prev Close (%)")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def build_post(label: str, pct: float | None, basis: str) -> str:
    stamp = pd.Timestamp.now(tz=TZ).strftime("%Y/%m/%d %H:%M")
    if pct is None:
        return f"▲ {label} 日中スナップショット（no data）\n+0.00%（基準: {basis}）\n#R_BANK9 #日本株\n"
    sign = "+" if pct >= 0 else ""
    return f"▲ {label} 日中スナップショット（{stamp} JST）\n{sign}{pct:.2f}%（基準: {basis}）\n#R_BANK9 #日本株\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--index-key", required=True)
    ap.add_argument("--basis", default="prev_close")
    ap.add_argument("--session-start", default="09:00")
    ap.add_argument("--session-end", default="15:30")
    ap.add_argument("--day-anchor", default="09:00")
    ap.add_argument("--value-type", default="auto")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-text", required=True)
    ap.add_argument("--snapshot-png", required=True)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_png = Path(args.snapshot_png)
    out_txt = Path(args.out_text)
    out_json = Path(args.out_json)

    df = read_ts_pct(csv_path)

    # 最新値（なければ None）
    lp = latest_pct(df)

    # チャート
    render_chart(df, out_png, args.label)

    # ポスト文
    post = build_post(args.label, lp, args.basis)
    out_txt.write_text(post, encoding="utf-8")

    # stats.json
    out_json.write_text(
        json.dumps(
            {
                "index_key": args.index_key,
                "label": args.label,
                "pct_intraday": 0.0 if lp is None else lp,
                "basis": args.basis,
                "session": {"start": args.session_start, "end": args.session_end, "anchor": args.day_anchor},
                "updated_at": pd.Timestamp.now(tz=TZ).isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

if __name__ == "__main__":
    main()
