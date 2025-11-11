#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
R-BANK9 intraday snapshot/post generator (robust)

優先順位:
  1) intraday CSV (ts,pct) から時系列を描画し、最新値を採用
  2) intraday が空なら history CSV (date,value) の『当日ぶん』を pct としてフォールバック
  3) それも無ければ 0.00% / no data

出力:
  - site_intraday.png  … ダークテーマ・縁なし
  - site_post.txt      … フォールバック時は（最終値）を注記
  - site_stats.json    … pct_intraday を反映
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt

TZ = "Asia/Tokyo"

# ====== ダークテーマ（縁なし） ======
BG = "#0b1220"      # 背景
FG = "#d1d5db"      # 文字色
GRID = "#334155"    # グリッド

def _set_dark_matplotlib():
    plt.rcParams["figure.facecolor"] = BG
    plt.rcParams["axes.facecolor"] = BG
    plt.rcParams["savefig.facecolor"] = BG
    plt.rcParams["savefig.edgecolor"] = BG

def _style_axes(ax):
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(colors=FG, labelcolor=FG)
    ax.grid(True, alpha=0.35, color=GRID, linestyle="-", linewidth=0.7)
    ax.title.set_color(FG)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)

# ---------------- Intraday CSV 読み ----------------
def read_ts_pct(csv_path: Path) -> pd.DataFrame:
    """
    intraday CSV をラフにクレンジングして ts, pct を返す。
    - コメント行(#...), 空行, カンマ数≠1 行を除外
    - ヘッダの有無は問わない（header=None で読み、型変換時に NaT/NaN を落とす）
    """
    if not csv_path.exists():
        return pd.DataFrame(columns=["ts", "pct"])

    raw = csv_path.read_text(encoding="utf-8", errors="ignore")
    lines = []
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.count(",") != 1:
            continue
        # 例: "ts,pct" のヘッダも混ざって良い（後で NaT/NaN で落ちる）
        lines.append(s)

    if not lines:
        return pd.DataFrame(columns=["ts", "pct"])

    buf = io.StringIO("\n".join(lines))
    try:
        df = pd.read_csv(buf, header=None, names=["ts", "pct"])
    except Exception:
        return pd.DataFrame(columns=["ts", "pct"])

    # 型変換（タイムゾーンは JST に統一）
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True).dt.tz_convert(TZ)
    df["pct"] = pd.to_numeric(df["pct"], errors="coerce")
    df = df.dropna(subset=["ts", "pct"]).sort_values("ts").reset_index(drop=True)
    return df

def latest_pct_from_intraday(df: pd.DataFrame) -> Optional[float]:
    if df is None or df.empty:
        return None
    return float(df["pct"].iloc[-1])

# ---------------- History CSV フォールバック ----------------
def read_history_pct(history_csv: Path) -> Optional[float]:
    """
    history CSV (date,value) を読み、当日分の value を pct として返す。
    値が当日分で見つからなければ None。
    """
    if not history_csv or not history_csv.exists():
        return None

    try:
        h = pd.read_csv(history_csv)
    except Exception:
        return None

    # 列名ゆらぎ対策
    cols = {c.lower().strip(): c for c in h.columns}
    if "date" not in cols or "value" not in cols:
        return None

    h = h.rename(columns={cols["date"]: "date", cols["value"]: "value"})
    h["date"] = pd.to_datetime(h["date"], errors="coerce").dt.tz_localize(None)
    h["value"] = pd.to_numeric(h["value"], errors="coerce")

    h = h.dropna(subset=["date", "value"]).sort_values("date")
    if h.empty:
        return None

    today = pd.Timestamp.now(tz=TZ).date()
    ht = h[h["date"].dt.date == today]
    if ht.empty:
        return None
    return float(ht.iloc[-1]["value"])

# ---------------- 描画 ----------------
def render_chart(df: pd.DataFrame, out_png: Path, label: str):
    _set_dark_matplotlib()
    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    _style_axes(ax)

    title = f"{label} Intraday Snapshot (JST)"
    if df.empty:
        ax.set_title(title + " (no data)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Change vs Prev Close (%)")
        # 線は描かない
    else:
        ax.plot(df["ts"], df["pct"], linewidth=1.8, color="#f87171")  # 赤系（視認性）
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Change vs Prev Close (%)")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    # 縁なし保存
    fig.savefig(out_png, bbox_inches="tight", facecolor=BG, edgecolor=BG)
    plt.close(fig)

# ---------------- ポスト文 ----------------
def build_post(label: str, pct: Optional[float], basis: str, used_fallback: bool) -> str:
    stamp = pd.Timestamp.now(tz=TZ).strftime("%Y/%m/%d %H:%M")
    if pct is None:
        return f"▲ {label} 日中スナップショット（no data）\n+0.00%（基準: {basis}）\n#R_BANK9 #日本株\n"
    sign = "+" if pct >= 0 else ""
    suffix = "（最終値）" if used_fallback else ""
    return f"▲ {label} 日中スナップショット（{stamp} JST）{suffix}\n{sign}{pct:.2f}%（基準: {basis}）\n#R_BANK9 #日本株\n"

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)                  # intraday
    ap.add_argument("--history-csv", required=False)         # history (fallback)
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
    hist_path = Path(args.history_csv) if args.history_csv else None
    out_png  = Path(args.snapshot_png)
    out_txt  = Path(args.out_text)
    out_json = Path(args.out_json)

    # 1) intraday
    df = read_ts_pct(csv_path)
    lp = latest_pct_from_intraday(df)

    used_fallback = False
    if lp is None and hist_path:
        # 2) history フォールバック
        hp = read_history_pct(hist_path)
        if hp is not None:
            lp = hp
            used_fallback = True

    # 画像
    render_chart(df, out_png, args.label)

    # テキスト
    post = build_post(args.label, lp, args.basis, used_fallback)
    out_txt.write_text(post, encoding="utf-8")

    # JSON
    out_json.write_text(
        json.dumps(
            {
                "index_key": args.index_key,
                "label": args.label,
                "pct_intraday": 0.0 if lp is None else lp,
                "basis": args.basis,
                "session": {
                    "start": args.session_start,
                    "end": args.session_end,
                    "anchor": args.day_anchor,
                },
                "updated_at": pd.Timestamp.now(tz=TZ).isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

if __name__ == "__main__":
    main()
