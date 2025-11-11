#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
R-BANK9 intraday snapshot/post generator (robust)
- 入力 CSV は rbank9_intraday.csv （推奨: ヘッダ ts,pct / 値は % ）
- コメント行(#...)・ゴミ行・空行を無視して読み取る
- 可能なら履歴CSV（rbank9_history.csv）でフォールバック（%）を補う
- 出力:
  * site_intraday.png
  * site_post.txt
  * site_stats.json
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

# ====== ダークテーマ（枠線なし） ======
BG   = "#0b1220"   # 背景
FG   = "#d1d5db"   # 文字色
GRID = "#334155"   # グリッド

def set_dark_axes(ax):
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_visible(False)     # 枠線消去
    ax.tick_params(colors=FG, labelcolor=FG)
    ax.grid(True, alpha=0.35, color=GRID, linestyle="-", linewidth=0.7)
    ax.title.set_color(FG)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)

def _clean_lines_for_ts_pct(raw: str) -> list[str]:
    cleaned = []
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        # ts,pct の 2 要素だけを通す（カンマ1個）
        if s.count(",") != 1:
            continue
        cleaned.append(s)
    return cleaned

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
    cleaned_lines = _clean_lines_for_ts_pct(raw)
    if not cleaned_lines:
        return pd.DataFrame(columns=["ts", "pct"])

    buf = io.StringIO("\n".join(cleaned_lines))
    # ヘッダが ts,pct の場合も / 値だけの場合も吸収する
    try:
        df_try = pd.read_csv(buf, header=None)
        if df_try.shape[1] == 2:
            df_try.columns = ["ts", "pct"]
        else:
            return pd.DataFrame(columns=["ts", "pct"])
    except Exception:
        return pd.DataFrame(columns=["ts", "pct"])

    # 文字列→型
    df_try["ts"]  = pd.to_datetime(df_try["ts"], errors="coerce", utc=True)
    df_try["ts"]  = df_try["ts"].dt.tz_convert(TZ)   # +09:00 を尊重してJSTへ
    df_try["pct"] = pd.to_numeric(df_try["pct"], errors="coerce")

    df_try = df_try.dropna(subset=["ts", "pct"]).sort_values("ts").reset_index(drop=True)
    return df_try

def read_history_pct(history_csv: Path) -> Optional[float]:
    """
    docs/outputs/rbank9_history.csv から直近値を % として読むフォールバック。
    期待列: date,value もしくは 2 列だけの CSV（第2列が数値）
    """
    if not history_csv.exists():
        return None
    try:
        txt = history_csv.read_text(encoding="utf-8", errors="ignore").strip()
        if not txt:
            return None
        # コメントや空行を除く
        lines = [l.strip() for l in txt.splitlines() if l.strip() and not l.strip().startswith("#")]
        if not lines:
            return None
        # カンマ区切り2列だけを残す
        pure = [l for l in lines if l.count(",") >= 1]
        if not pure:
            return None
        df = pd.read_csv(io.StringIO("\n".join(pure)))
        # 列名吸収
        cols = [c.lower().strip() for c in df.columns]
        if len(cols) >= 2:
            # 値列を第2列とみなす
            vcol = df.columns[1]
        else:
            return None
        s = pd.to_numeric(df[vcol], errors="coerce")
        s = s.dropna()
        if s.empty:
            return None
        return float(s.iloc[-1])
    except Exception:
        return None

def latest_pct(df: pd.DataFrame) -> Optional[float]:
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
        # 何も描かない（余計な線・枠が出ない）
        ax.set_title(title + " (no data)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Change vs Prev Close (%)")
    else:
        ax.plot(df["ts"], df["pct"], linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Change vs Prev Close (%)")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def build_post(label: str, pct: Optional[float], basis: str, used_fallback: bool) -> str:
    stamp = pd.Timestamp.now(tz=TZ).strftime("%Y/%m/%d %H:%M")
    if pct is None:
        return f"▲ {label} 日中スナップショット（no data）\n+0.00%（基準: {basis}）\n#R_BANK9 #日本株\n"
    sign = "+" if pct >= 0 else ""
    note = "（履歴値）" if used_fallback else ""
    return f"▲ {label} 日中スナップショット（{stamp} JST）{note}\n{sign}{pct:.2f}%（基準: {basis}）\n#R_BANK9 #日本株\n"

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
    out_png  = Path(args.snapshot_png)
    out_txt  = Path(args.out_text)
    out_json = Path(args.out_json)

    # 1) intraday(ts,pct) を読む
    df = read_ts_pct(csv_path)

    # 2) 最新値
    lp = latest_pct(df)
    used_fallback = False

    # 3) フォールバック（intraday 空のときのみ）
    if lp is None:
        # 同ディレクトリの履歴CSVを試す
        hist_path = csv_path.with_name("rbank9_history.csv")
        h = read_history_pct(hist_path)
        if h is not None:
            lp = h
            used_fallback = True

    # 4) チャート（no data なら線は描かない）
    render_chart(df, out_png, args.label)

    # 5) ポスト文
    post = build_post(args.label, lp, args.basis, used_fallback)
    out_txt.write_text(post, encoding="utf-8")

    # 6) stats.json
    out_json.write_text(
        json.dumps(
            {
                "index_key": args.index_key,
                "label": args.label,
                "pct_intraday": 0.0 if lp is None else lp,
                "basis": args.basis,
                "session": {
                    "start": args.session_start,
                    "end":   args.session_end,
                    "anchor": args.day_anchor
                },
                "used_fallback": used_fallback,
                "updated_at": pd.Timestamp.now(tz=TZ).isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

if __name__ == "__main__":
    main()
