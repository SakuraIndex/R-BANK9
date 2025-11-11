#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
R-BANK9 intraday snapshot/post generator (最終完全版)

入力 CSV (どちらでも可):
  A) 縦持ち: ヘッダ ts,pct（例: 2025-11-11T09:05:00+09:00,0.12）
  B) 横持ち: 1行目=ティッカー/コメント, 以降は 1列目=時刻, 2列目以降=各銘柄%（等加重平均を内部計算）

出力:
  - docs/outputs/rbank9_intraday.png   (ダークテーマ)
  - docs/outputs/rbank9_post_intraday.txt
  - docs/outputs/rbank9_stats.json
"""

from __future__ import annotations
from pathlib import Path
import argparse
import json
import re
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
    # 互換（無視）
    p.add_argument("--index-key", default=None)
    p.add_argument("--label", default=None)
    p.add_argument("--dt-col", default=None)
    p.add_argument("--value-type", default=None)
    p.add_argument("--basis", default=None)
    p.add_argument("--session-start", default=None)
    p.add_argument("--session-end", default=None)
    p.add_argument("--day-anchor", default=None)
    return p.parse_args()

# ---------- 表示スタイル ----------
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

# ---------- CSV 読み取り（A: 縦持ち） ----------
def _read_vertical(csv_path: Path) -> pd.DataFrame | None:
    df = pd.read_csv(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}
    if "ts" not in cols or "pct" not in cols:
        return None
    df = df.rename(columns={cols["ts"]: "ts", cols["pct"]: "pct"})
    df["ts"]  = pd.to_datetime(df["ts"], errors="coerce", utc=True).dt.tz_convert("Asia/Tokyo")
    df["pct"] = pd.to_numeric(df["pct"], errors="coerce")
    df = df.dropna(subset=["ts", "pct"]).sort_values("ts")
    if df.empty:
        return None
    return df[["ts","pct"]]

# ---------- CSV 読み取り（B: 横持ち → 等加重平均） ----------
def _read_wide(csv_path: Path) -> pd.DataFrame | None:
    raw = pd.read_csv(csv_path, header=None, dtype=str)
    if raw.empty or raw.shape[0] < 2:
        return None

    # 1行目に ".T" や '#' が多いなら横持ちと判断
    first_row = raw.iloc[0].astype(str).fillna("")
    if not any(x.endswith(".T") or x.strip().startswith("#") for x in first_row):
        return None

    # 2行目以降: 1列目=時刻、以降=各銘柄%（文字を数値化）
    data = raw.iloc[1:].reset_index(drop=True)
    # 時刻列
    ts = pd.to_datetime(data.iloc[:, 0], errors="coerce", utc=True).dt.tz_convert("Asia/Tokyo")

    # 数値列をまとめて等加重平均
    if data.shape[1] < 2:
        return None
    numeric_part = []
    for i in range(1, data.shape[1]):
        s = pd.to_numeric(data.iloc[:, i].str.replace(" ", "", regex=False), errors="coerce")
        numeric_part.append(s)

    if not numeric_part:
        return None

    import numpy as np
    mat = np.vstack([s.to_numpy() for s in numeric_part]).T  # shape: (rows, ncols)
    pct = np.nanmean(mat, axis=1)

    out = pd.DataFrame({"ts": ts, "pct": pct})
    out = out.dropna(subset=["ts"]).sort_values("ts")
    if out.empty:
        return None
    return out

# ---------- 総合 CSV 読み取り ----------
def read_csv_any(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise ValueError(f"CSV がありません: {csv_path}")

    # BOM/空行除去（安全化）
    txt = csv_path.read_text(encoding="utf-8").replace("\ufeff", "")
    cleaned = "\n".join(line.strip() for line in txt.splitlines() if line.strip())
    if not cleaned:
        raise ValueError("CSV が空です。")
    csv_path.write_text(cleaned, encoding="utf-8")

    # A) ts,pct を試す
    df = _read_vertical(csv_path)
    if df is not None:
        return df

    # B) 横持ち → 等加重平均を試す
    df = _read_wide(csv_path)
    if df is not None:
        return df

    # どちらでもなければ詳細を出して失敗
    probe = pd.read_csv(csv_path, nrows=1, header=None)
    raise ValueError(f"CSV 形式を解釈できません。先頭行={probe.iloc[0].tolist()}")

# ---------- 描画 ----------
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

# ---------- テキスト/JSON ----------
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

# ---------- main ----------
def main():
    args = parse_args()

    in_csv   = Path(args.csv) if args.csv else IN_CSV_DEF
    out_png  = Path(args.snapshot_png) if args.snapshot_png else OUT_PNG_DEF  # ← argparse は '-' を '_' に変換
    out_txt  = Path(args.out_text) if args.out_text else OUT_TXT_DEF
    out_stat = Path(args.out_json) if args.out_json else OUT_STAT_DEF

    index_key = (args.index_key or INDEX_KEY_DEF)
    label     = (args.label or LABEL_DEF)

    df = read_csv_any(in_csv)
    latest_pct = float(df["pct"].iloc[-1])

    plot_series(df, out_png)

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text(build_post_text(latest_pct, label), encoding="utf-8")

    out_stat.parent.mkdir(parents=True, exist_ok=True)
    out_stat.write_text(build_stats_json(latest_pct, index_key, label), encoding="utf-8")

    print(f"[ok] rows={len(df)} latest={latest_pct:.4f} -> {out_png}, {out_txt}, {out_stat}")

if __name__ == "__main__":
    main()
