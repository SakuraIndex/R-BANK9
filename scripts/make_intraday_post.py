#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
R-BANK9 intraday snapshot/post generator 〔完全版〕
入力 : docs/outputs/rbank9_intraday.csv
       ヘッダは ts,pct
       ts は "YYYY-MM-DDTHH:MM(+09:00)" または "YYYY-MM-DDTHH:MM:SS(+09:00)"
出力 :
  - docs/outputs/rbank9_intraday.png
  - docs/outputs/rbank9_post_intraday.txt
  - docs/outputs/rbank9_stats.json
"""

from __future__ import annotations
from pathlib import Path
import argparse
import json
import re
from typing import Tuple

import pandas as pd
import matplotlib.pyplot as plt

# ===== デフォルト入出力（引数で上書き可） =====
IN_CSV   = Path("docs/outputs/rbank9_intraday.csv")
OUT_PNG  = Path("docs/outputs/rbank9_intraday.png")
OUT_TXT  = Path("docs/outputs/rbank9_post_intraday.txt")
OUT_STAT = Path("docs/outputs/rbank9_stats.json")

TITLE = "R-BANK9 Intraday Snapshot (JST)"

# ===== Util =====
def _ensure_seconds(ts: str) -> str:
    """'YYYY-MM-DDTHH:MM+09:00' のように秒が無い場合に ':00' を補う。"""
    if not isinstance(ts, str):
        return ts
    # 末尾が +09:00 / -09:00 / Z などの前に :SS が無ければ追加
    m = re.match(r"^(.+T\d{2}:\d{2})(?!(?::\d{2}))(\+|\-|Z)", ts)
    if m:
        return ts.replace(m.group(1), m.group(1) + ":00")
    return ts

def _read_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise ValueError(f"CSV がありません: {csv_path}")

    raw = csv_path.read_text(encoding="utf-8").replace("\ufeff", "")
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("CSV が空です。")

    # 余計なインラインコメントや全角カンマ等を排除して再保存（頑健化）
    cleaned = []
    for ln in lines:
        # CSV で無い行（例: 説明やヘッダの装飾など）を弾く
        if "," not in ln:
            continue
        cleaned.append(ln)
    csv_path.write_text("\n".join(cleaned) + "\n", encoding="utf-8")

    df = pd.read_csv(csv_path)

    # 列名ゆらぎ吸収（Datetime/TS/ts, pct/percent など）
    colmap = {c.lower().strip(): c for c in df.columns}
    ts_col = next((colmap[k] for k in ("ts", "datetime", "time")), None)
    pct_col = next((colmap[k] for k in ("pct", "percent", "change_pct")), None)
    if not ts_col or not pct_col:
        raise ValueError(f"必要列が見つかりません（ts と pct）。got={list(df.columns)}")

    df = df.rename(columns={ts_col: "ts", pct_col: "pct"})

    # 秒なし→追加
    df["ts"] = df["ts"].astype(str).map(_ensure_seconds)

    # 例: 2025-11-11T10:00:00+09:00 / 2025-11-11T10:00:00Z 両対応
    # 明示フォーマットで高速・確実に
    parsed = pd.to_datetime(
        df["ts"],
        format="%Y-%m-%dT%H:%M:%S%z",
        errors="coerce",
    )
    # もし format 不一致が多ければフォールバック（dateutil）
    if parsed.isna().all():
        parsed = pd.to_datetime(df["ts"], errors="coerce", utc=True)

    df["ts"] = parsed.dt.tz_convert("Asia/Tokyo")
    # pct は数値へ
    df["pct"] = pd.to_numeric(df["pct"], errors="coerce")

    df = df.dropna(subset=["ts", "pct"]).sort_values("ts")
    # 同一時刻の重複は最後を採用
    df = df.drop_duplicates(subset=["ts"], keep="last")
    return df

def _setup_dark(ax):
    # ダークテーマ
    fig = ax.figure
    fig.patch.set_facecolor("#0b1420")
    ax.set_facecolor("#0b1420")
    for spine in ax.spines.values():
        spine.set_color("#89a2b8")
    ax.tick_params(colors="#d4e9f7")
    ax.xaxis.label.set_color("#cfe6f3")
    ax.yaxis.label.set_color("#cfe6f3")
    ax.title.set_color("#e7f3ff")
    ax.grid(True, alpha=0.25, color="#2a4158")

def _plot(df: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    _setup_dark(ax)

    if df.empty:
        ax.set_title(TITLE)
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color="#cfe6f3", fontsize=16)
        ax.set_xlabel("Time")
        ax.set_ylabel("Change vs Prev Close (%)")
    else:
        ax.plot(df["ts"], df["pct"], linewidth=2.0)
        ax.set_title(TITLE)
        ax.set_xlabel("Time")
        ax.set_ylabel("Change vs Prev Close (%)")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def _write_post_and_stats(latest_pct: float, out_txt: Path, out_stat: Path) -> None:
    now = pd.Timestamp.now(tz="Asia/Tokyo")
    sign = "+" if latest_pct >= 0 else ""
    post = (
        f"▲ R-BANK9 日中スナップショット（{now.strftime('%Y/%m/%d %H:%M')} JST）\n"
        f"{sign}{latest_pct:.2f}%（基準: prev_close）\n"
        f"#R_BANK9 #日本株\n"
    )
    out_txt.write_text(post, encoding="utf-8")

    stat = {
        "index_key": "R_BANK9",
        "label": "R-BANK9",
        "pct_intraday": float(round(latest_pct, 6)),
        "basis": "prev_close",
        "session": {"start": "09:00", "end": "15:30", "anchor": "09:00"},
        "updated_at": now.isoformat(),
    }
    out_stat.write_text(json.dumps(stat, ensure_ascii=False, indent=2), encoding="utf-8")

def _write_fallback(out_png: Path, out_txt: Path, out_stat: Path, reason: str) -> None:
    # 失敗時の見た目（ダークで no data）
    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    _setup_dark(ax)
    ax.set_title(TITLE)
    ax.text(0.5, 0.5, "no data", ha="center", va="center",
            transform=ax.transAxes, color="#cfe6f3", fontsize=16)
    ax.set_xlabel("Time")
    ax.set_ylabel("Change vs Prev Close (%)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    now = pd.Timestamp.now(tz="Asia/Tokyo")
    out_txt.write_text(
        "▲ R-BANK9 日中スナップショット（no data）\n"
        "+0.00%（基準: prev_close）\n"
        "#R_BANK9 #日本株\n", encoding="utf-8"
    )
    out_stat.write_text(json.dumps({
        "index_key": "R_BANK9",
        "label": "R-BANK9",
        "pct_intraday": 0.0,
        "basis": "prev_close",
        "updated_at": now.isoformat(),
        "note": f"fallback: {reason}",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=str(IN_CSV))
    p.add_argument("--out-json", default=str(OUT_STAT))
    p.add_argument("--out-text", default=str(OUT_TXT))
    p.add_argument("--snapshot-png", default=str(OUT_PNG))
    return p.parse_args()

def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    out_png  = Path(args.snapshot_png)
    out_txt  = Path(args.out_text)
    out_stat = Path(args.out_json)

    try:
        df = _read_csv(csv_path)
    except Exception as e:
        # ここで「no data 画像」を作るが、公開までは止めたいので exit 1
        _write_fallback(out_png, out_txt, out_stat, reason=str(e))
        print(f"[error] CSV read failed: {e}")
        raise SystemExit(1)

    # データは読めたが空 → no data 扱い（exit 1 で公開停止）
    if df.empty:
        _write_fallback(out_png, out_txt, out_stat, reason="empty dataframe after parse")
        print("[error] dataframe is empty after parsing")
        raise SystemExit(1)

    latest_pct = float(df["pct"].iloc[-1])

    # 画像
    _plot(df, out_png)
    # 投稿文 & stats
    _write_post_and_stats(latest_pct, out_txt, out_stat)

if __name__ == "__main__":
    main()
