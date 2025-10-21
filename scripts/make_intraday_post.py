#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
R-BANK9 / ASTRA4 など共通で使う、日中スナップショット作成スクリプト 最終版

主なポイント
- 値種別: --value-type {ratio,percent} を厳格化
  - percent 指定時はプロット・文言とも % で統一（必要に応じて *100）
  - 誤爆防止のヒューリスティクス付き（|max| < 5 のとき ratio とみなして *100）
- 日時列: tz-naive は Asia/Tokyo で localize、tz-aware は Asia/Tokyo へ convert
- CSV 日時列名: --dt-col で明示（例: "Unnamed: 0"）… YAML 側ではクォート必須
- 例外/NaN/Inf 安全化: ±inf→NaN、dropna()、最終値が無い時は丁寧にエラー
- PNG: 黒背景・白縁なし・tight・スパイン非表示・余白ゼロ
- JSON: pandas.io.json は使用せず、標準 json で保存
- ログを詳細出力（csvの形状、列名、採用した value_type、保存先など）
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

JST = timezone.utc  # 仮置き（後で tz_convert で Asia/Tokyo を使う）
ASIA_TOKYO = "Asia/Tokyo"


# ---------- ユーティリティ ----------

def to_jst_index(df: pd.DataFrame, dt_col: str) -> pd.DataFrame:
    """CSVの日時列 dt_col を index にし、Asia/Tokyo の tz-aware DatetimeIndex に整形。"""
    if dt_col not in df.columns:
        raise ValueError(f"CSVに対象列 '{dt_col}' がありません。列={list(df.columns)}")

    dt = pd.to_datetime(df[dt_col], errors="coerce", infer_datetime_format=True)
    if dt.isna().all():
        raise ValueError(f"Datetime列 '{dt_col}' が全て欠損です。")

    # tz-naive → localize('Asia/Tokyo') / tz-aware → tz_convert('Asia/Tokyo')
    if dt.dt.tz is None:
        dt = dt.dt.tz_localize(ASIA_TOKYO)
    else:
        dt = dt.dt.tz_convert(ASIA_TOKYO)

    out = df.drop(columns=[dt_col]).copy()
    out.index = dt
    out = out.sort_index()
    return out


def decide_percent_series(series: pd.Series, value_type: str) -> Tuple[pd.Series, str]:
    """
    入力 series をプロット/出力用に整形。
    - value_type == 'percent' なら基本は %。
    - ただし、誤って ratio(0.003 など)が入ってきた場合に備えて、|max|<5 のとき *100。
    戻り値: (整形済みシリーズ, y軸ラベル)
    """
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        raise ValueError("対象シリーズに有効な値がありません。")

    if value_type.lower() == "percent":
        # すでに % の可能性と比率(ratio)の可能性の両方に耐える
        max_abs = float(np.nanmax(np.abs(s.values)))
        if max_abs < 5.0:  # 5%未満なら ratio（0.05=5% 未満）とみなして *100
            s = s * 100.0
        ylab = "Change vs Prev Close (%)"
    else:
        # ratio はそのまま（ただし見やすさのため % 表示に統一する設計にも対応可能）
        # 今回は ratio 指定時でもチャート/テキストは % で出すよう統一（＝ *100）
        s = s * 100.0
        ylab = "Change vs Prev Close (%)"

    return s, ylab


def latest_pct_text(pct: float) -> str:
    """数値→ ±x.xx% のテキスト（矢印つき見出し用の ▲/▼ も別途生成）。"""
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.2f}%"


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def style_dark_ax(ax: plt.Axes) -> None:
    # 背景ダーク・スパイン/余白なし
    ax.set_facecolor("black")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors="white", labelsize=9)
    ax.title.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")


# ---------- メイン処理 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-key", required=True, help="出力ラベルとJSONに入れるキー（例: R_BANK9 / ASTRA4）")
    ap.add_argument("--csv", required=True, help="入力CSVパス")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-text", required=True)
    ap.add_argument("--snapshot-png", required=True)
    ap.add_argument("--session-start", required=True)   # "09:00"
    ap.add_argument("--session-end", required=True)     # "15:30"
    ap.add_argument("--day-anchor", required=True)      # "09:00"
    ap.add_argument("--basis", default="prev_close")
    ap.add_argument("--value-type", choices=["ratio", "percent"], default="percent")
    ap.add_argument("--dt-col", required=True, help="日時列名（例: 'Datetime' / 'Unnamed: 0'）")
    ap.add_argument("--label", default=None, help="プロット凡例に使うラベル（未指定はindex-key）")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_json = Path(args.out_json)
    out_text = Path(args.out_text)
    out_png = Path(args.snapshot_png)

    print("=== Generate intraday snapshot ===")
    print(f"VALUE_TYPE={args.value_type} (default=percent)")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    raw = pd.read_csv(csv_path)
    print(f"[make_intraday_post] CSV file: {csv_path}, rows={len(raw)}, cols={list(raw.columns)}")

    df = to_jst_index(raw, args.dt_col)
    print(f"[make_intraday_post] Datetime column = '{args.dt_col}', rows = {len(df)} (JST)")

    # 対象列は index-key と同名にする運用（例: 'R_BANK9', 'ASTRA4'）
    # 無ければ、列名を全部出してエラー
    tgt_col = args.index_key
    print(f"[make_intraday_post] using column = '{tgt_col}'")
    if tgt_col not in df.columns:
        raise ValueError(f"CSV に対象列 '{tgt_col}' がありません。列={list(df.columns)}")

    series = df[tgt_col].copy()
    series, ylab = decide_percent_series(series, args.value_type)
    print(f"[make_intraday_post] decided value_type = {args.value_type}")

    # 描画
    lbl = args.label or args.index_key.replace("_", "-")
    fig, ax = plt.subplots(figsize=(11.5, 6.0), facecolor="black")
    style_dark_ax(ax)
    ax.plot(series.index, series.values, linewidth=2.0, color="#00E5FF", label=lbl)
    ax.set_ylabel(ylab)
    ax.set_xlabel("Time")
    # 直近日時（JST）でタイトル
    now_jst = pd.Timestamp.now(tz=ASIA_TOKYO)
    ax.set_title(f"{lbl} Intraday Snapshot ({now_jst:%Y/%m/%d %H:%M})")

    # グリッドは薄めに
    ax.grid(True, linestyle="-", alpha=0.25, color="white")
    ax.legend(facecolor="black", edgecolor="none", labelcolor="white")

    # 白縁/余白なしで保存
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=0.2)
    fig.savefig(
        out_png,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        edgecolor="none",
        dpi=160,
    )
    plt.close(fig)
    print(f"[make_intraday_post] snapshot saved -> {out_png}")

    # テキスト出力（現在値）
    last_pct = float(series.dropna().iloc[-1])
    sign_head = "▲" if last_pct >= 0 else "▼"
    line1 = f"{sign_head} {lbl} 日中スナップショット ({now_jst:%Y/%m/%d %H:%M})"
    line2 = f"{latest_pct_text(last_pct)}（基準: {args.basis}）"
    line3 = f"#{args.index_key.replace('_', '-') } #日本株"
    save_text(out_text, [line1, line2, line3])
    print(f"[make_intraday_post] text saved -> {out_text}")

    # JSON
    payload = {
        "index_key": args.index_key,
        "label": lbl,
        "pct_intraday": last_pct,
        "basis": args.basis,
        "session": {
            "start": args.session_start,
            "end": args.session_end,
            "anchor": args.day_anchor,
        },
        "updated_at": now_jst.isoformat(),
    }
    save_json(out_json, payload)
    print(f"[make_intraday_post] json saved -> {out_json}")


if __name__ == "__main__":
    main()
