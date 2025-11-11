#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
R-BANK9 intraday snapshot/post generator (robust)
- 入力: docs/outputs/rbank9_intraday.csv
    * 2列形式:   ts,pct
    * 多列形式:  (先頭が時刻列) + 各銘柄列 + 末尾に R_BANK9 列（あればそれを使用）
      ヘッダに "# ..." のコメント/全角/空ヘッダも吸収
- 出力:
  - docs/outputs/rbank9_intraday.png
  - docs/outputs/rbank9_post_intraday.txt
  - docs/outputs/rbank9_stats.json
"""

from __future__ import annotations
from pathlib import Path
import argparse
import json
import re
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt

# デフォルトパス（CLI で上書き可能）
DEF_IN_CSV   = Path("docs/outputs/rbank9_intraday.csv")
DEF_OUT_PNG  = Path("docs/outputs/rbank9_intraday.png")
DEF_OUT_TXT  = Path("docs/outputs/rbank9_post_intraday.txt")
DEF_OUT_STAT = Path("docs/outputs/rbank9_stats.json")

TITLE = "R-BANK9 Intraday Snapshot (JST)"

# ---- dark style ----
def set_dark_style():
    plt.rcParams.update({
        "figure.facecolor": "#0b1420",
        "axes.facecolor":   "#0f1b28",
        "savefig.facecolor":"#0b1420",
        "axes.edgecolor":   "#294054",
        "axes.labelcolor":  "#d4e9f7",
        "xtick.color":      "#cfe6f3",
        "ytick.color":      "#cfe6f3",
        "text.color":       "#d4e9f7",
        "grid.color":       "#2b3e52",
        "axes.titleweight": "bold",
    })

# ---- helpers ----
def _read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"CSV がありません: {path}")
    t = path.read_text(encoding="utf-8").replace("\ufeff", "")
    return "\n".join([ln.rstrip() for ln in t.splitlines() if ln.strip()])

def _clean_headers(cols) -> list[str]:
    clean = []
    for c in cols:
        s = str(c)
        s = re.sub(r"\s*#.*$", "", s)  # コメント除去
        s = s.replace("　", " ").strip()
        if s == "":
            s = "ts"                  # 空ヘッダは時刻列と仮定
        clean.append(s)
    return clean

def _load_csv_any(csv_path: Path, dt_col_hint: str) -> pd.DataFrame:
    """可能な限り頑健に ts/pct の DataFrame を返す。"""
    raw = _read_text(csv_path)
    csv_path.write_text(raw, encoding="utf-8")  # pandas 用にクリーン版を書き戻し

    df = pd.read_csv(csv_path, engine="python")
    df.columns = _clean_headers(df.columns)

    # ts 列候補
    ts_candidates = [dt_col_hint, "ts", "datetime", "date", "time", "Timestamp", "Unnamed: 0"]
    ts_name = next((c for c in ts_candidates if c in df.columns), None)
    if ts_name is None:
        ts_name = df.columns[0]  # 先頭列を ts と見なす

    # 値列の決定
    val_col = "pct" if "pct" in df.columns else ("R_BANK9" if "R_BANK9" in df.columns else None)

    ts = pd.to_datetime(df[ts_name], errors="coerce", utc=True)
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    ts = ts.dt.tz_convert("Asia/Tokyo")

    if val_col is not None:
        val = pd.to_numeric(df[val_col], errors="coerce")
        out = pd.DataFrame({"ts": ts, "pct": val})
    else:
        # ts 以外の数値列を等ウェイト平均
        numeric_cols = []
        for c in df.columns:
            if c == ts_name:
                continue
            v = pd.to_numeric(df[c], errors="coerce")
            if v.notna().any():
                numeric_cols.append(c)
                df[c] = v
        if numeric_cols:
            out = pd.DataFrame({"ts": ts, "pct": df[numeric_cols].mean(axis=1, skipna=True)})
        else:
            out = pd.DataFrame({"ts": ts, "pct": pd.Series([], dtype=float)})

    return out.dropna(subset=["ts", "pct"]).sort_values("ts")

def _apply_unit(series: pd.Series, value_type: str) -> pd.Series:
    """
    value_type:
      - 'percent' : 値をそのまま % として扱う
      - 'ratio'   : 比を % に変換（*100）
      - 'auto'    : |中央値|<=1.2 なら % とみなし、それ以外は *100
    """
    s = pd.to_numeric(series, errors="coerce")
    if value_type == "percent":
        return s
    if value_type == "ratio":
        return s * 100.0
    med = s.abs().median()
    return s if (pd.notna(med) and med <= 1.2) else s * 100.0

def _plot_no_data(out_png: Path):
    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    ax.set_title(TITLE)
    ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, fontsize=18)
    ax.set_xlabel("Time")
    ax.set_ylabel("Change vs Prev Close (%)")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def _plot_series(df: pd.DataFrame, out_png: Path):
    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    if df.empty:
        ax.set_title(TITLE)
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, fontsize=18)
        ax.set_xlabel("Time")
        ax.set_ylabel("Change vs Prev Close (%)")
        ax.grid(True, alpha=0.35)
    else:
        ax.plot(df["ts"], df["pct"])
        ax.set_title(TITLE)
        ax.set_xlabel("Time")
        ax.set_ylabel("Change vs Prev Close (%)")
        ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index-key", required=True)
    p.add_argument("--label",      required=True)
    p.add_argument("--csv",        default=str(DEF_IN_CSV))
    p.add_argument("--dt-col",     default="ts")
    p.add_argument("--value-type", default="auto", choices=["auto","ratio","percent"])
    p.add_argument("--basis",      default="prev_close")
    p.add_argument("--session-start", default="09:00")
    p.add_argument("--session-end",   default="15:30")
    p.add_argument("--day-anchor",    default="09:00")
    p.add_argument("--out-json",      default=str(DEF_OUT_STAT))
    p.add_argument("--out-text",      default=str(DEF_OUT_TXT))
    p.add_argument("--snapshot-png",  default=str(DEF_OUT_PNG))
    args = p.parse_args()

    # ローカルパス（global は使わない）
    in_csv  = Path(args.csv)
    out_png = Path(args.snapshot_png)
    out_txt = Path(args.out_text)
    out_jsn = Path(args.out_json)

    set_dark_style()

    try:
        df = _load_csv_any(in_csv, args.dt_col)
        if not df.empty:
            df["pct"] = _apply_unit(df["pct"], args.value_type)
    except Exception as e:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        _plot_no_data(out_png)
        out_txt.write_text(
            f"▲ {args.label} 日中スナップショット（no data）\n+0.00%（基準: {args.basis}）\n#{args.index_key} #日本株\n",
            encoding="utf-8"
        )
        out_jsn.write_text(json.dumps({
            "index_key": args.index_key,
            "label": args.label,
            "pct_intraday": 0.0,
            "basis": args.basis,
            "session": {"start": args.session_start, "end": args.session_end, "anchor": args.day_anchor},
            "updated_at": pd.Timestamp.now(tz="Asia/Tokyo").isoformat()
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[warn] CSV parse failed: {e}")
        return

    latest_pct = float(df["pct"].iloc[-1]) if not df.empty else 0.0

    out_png.parent.mkdir(parents=True, exist_ok=True)
    _plot_series(df, out_png)

    sign = "+" if latest_pct >= 0 else ""
    out_txt.write_text(
        f"▲ {args.label} 日中スナップショット（{pd.Timestamp.now(tz='Asia/Tokyo').strftime('%Y/%m/%d %H:%M')} JST）\n"
        f"{sign}{latest_pct:.2f}%（基準: {args.basis}）\n"
        f"#{args.index_key} #日本株\n",
        encoding="utf-8"
    )

    out_jsn.write_text(json.dumps({
        "index_key": args.index_key,
        "label": args.label,
        "pct_intraday": latest_pct,
        "basis": args.basis,
        "session": {"start": args.session_start, "end": args.session_end, "anchor": args.day_anchor},
        "updated_at": pd.Timestamp.now(tz="Asia/Tokyo").isoformat()
    }, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
