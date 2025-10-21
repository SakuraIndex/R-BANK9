#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === ユーティリティ ===

JST = "Asia/Tokyo"


def to_jst_index(df: pd.DataFrame, dt_col: str) -> pd.DataFrame:
    """dt_col を DatetimeIndex(JST) にして返す。"""
    if dt_col not in df.columns:
        raise ValueError(f"Datetime column '{dt_col}' not found in CSV.")
    # pandas 2.0+ 警告対応: infer_datetime_format はデフォルト有効
    dt = pd.to_datetime(df[dt_col], errors="coerce", utc=False)
    # tz-naive → JST 付与
    if getattr(dt.dt, "tz", None) is None:
        dt = dt.dt.tz_localize(JST, nonexistent="shift_forward", ambiguous="NaT")
    else:
        dt = dt.dt.tz_convert(JST)
    df = df.drop(columns=[dt_col]).copy()
    df.index = dt
    df = df.sort_index()
    return df


def infer_csv_unit(s: pd.Series) -> str:
    """CSVの値が ratio(0.006)か percent(0.6)かをざっくり推定。"""
    v = pd.to_numeric(s, errors="coerce").dropna()
    if v.empty:
        return "ratio"
    p95 = v.quantile(0.95)
    # 95%点が 5 を超えるなら「％値が入っている」と判定（例: 12.3, 0.8 など）
    return "percent" if p95 > 5 else "ratio"


def normalize_to_ratio(s: pd.Series, csv_unit: str) -> pd.Series:
    """入力列 s を内部表現 ratio へ正規化。"""
    s = pd.to_numeric(s, errors="coerce")
    if csv_unit == "ratio":
        return s
    if csv_unit == "percent":
        return s / 100.0
    # auto
    unit = infer_csv_unit(s)
    return normalize_to_ratio(s, unit)


def pick_label_column(df: pd.DataFrame, label: str) -> pd.Series:
    """指数・凡例のラベルを y データ列名として使用。なければ最初の列を使う。"""
    if label in df.columns:
        return df[label]
    # 列名が一覧（コード等）のとき、単一系列を指す確実な方法がないため最左列を採る
    return df.iloc[:, 0]


def clip_session(
    df: pd.DataFrame, start_hhmm: str, end_hhmm: str
) -> Tuple[pd.DataFrame, Tuple[pd.Timestamp, pd.Timestamp]]:
    """JSTの HH:MM〜HH:MM で当日分を切り出す。"""
    # 当日の anchor は index の日付から推定（index が複数日の場合でも当日で切る）
    if df.index.tz is None:
        df.index = df.index.tz_localize(JST)
    jst = df.index.tz_localize(None)  # naive化（時刻だけ使うため）
    d0 = jst[-1].date() if len(jst) else pd.Timestamp.now(tz=JST).date()
    t0 = pd.to_datetime(f"{d0} {start_hhmm}").tz_localize(JST)
    t1 = pd.to_datetime(f"{d0} {end_hhmm}").tz_localize(JST)
    out = df[(df.index >= t0) & (df.index <= t1)]
    return out, (t0, t1)


def fig_dark_no_border(w=12, h=5):
    """黒ベース + 余白最小 + 枠線なしの Figure/Axes を返す。"""
    fig = plt.figure(figsize=(w, h), facecolor="black")
    ax = fig.add_axes([0.04, 0.08, 0.94, 0.86])  # ぴったりレイアウト
    ax.set_facecolor("black")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    return fig, ax


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="R-BANK9 intraday post & snapshot")
    p.add_argument("--index-key", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-text", required=True)
    p.add_argument("--snapshot-png", required=True)
    p.add_argument("--session-start", default="09:00")
    p.add_argument("--session-end", default="15:30")
    p.add_argument("--day-anchor", default="09:00")
    p.add_argument("--basis", default="prev_close")
    p.add_argument("--value-type", choices=["ratio", "percent"], default="percent")
    p.add_argument(
        "--csv-unit",
        choices=["auto", "ratio", "percent"],
        default="auto",
        help="単位 of CSV: ratio(0.006) or percent(0.6). auto=推定",
    )
    p.add_argument("--dt-col", default="Unnamed: 0")
    p.add_argument("--label", default="R-BANK9")
    return p


def main():
    args = build_parser().parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df_raw = pd.read_csv(csv_path)
    df = to_jst_index(df_raw, args.dt_col)

    # 取引時間で切り出し
    df_sess, (t0, t1) = clip_session(df, args.session_start, args.session_end)
    if df_sess.empty:
        raise ValueError(
            f"セッション内データがありません。範囲: {args.session_start}–{args.session_end} JST"
        )

    # 表示する系列を選択 → 入力単位を ratio へそろえる
    s_raw = pick_label_column(df_sess, args.label)
    s_ratio = normalize_to_ratio(s_raw, args.csv_unit)

    # 出力（見せ方）を作る
    if args.value_type == "percent":
        s_plot = s_ratio * 100.0
        y_label = "Change vs Prev Close (%)"
    else:
        s_plot = s_ratio
        y_label = "Change vs Prev Close (ratio)"

    # 最新％（小数）を JSON/テキストに採用
    pct_intraday = float((s_ratio.iloc[-1]) * 100.0)

    # === グラフ ===
    fig, ax = fig_dark_no_border()
    # 線色はデフォルト（白地に合わせた色を指定しない）
    ax.plot(s_plot.index, s_plot.values, linewidth=2.2)
    ax.set_title(
        f"{args.label} Intraday Snapshot ({pd.Timestamp.now(tz=JST):%Y/%m/%d %H:%M})",
        color="white",
        pad=10,
    )
    ax.set_xlabel("Time")
    ax.set_ylabel(y_label)
    ax.grid(color="#333333", linewidth=0.6)
    leg = ax.legend([args.label])
    for text in leg.get_texts():
        text.set_color("white")

    out_png = Path(args.snapshot_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, facecolor="black", bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)

    # === JSON ===
    payload = {
        "index_key": args.index_key,
        "label": args.label,
        "pct_intraday": pct_intraday,  # ％（小数）
        "basis": args.basis,
        "session": {
            "start": args.session_start,
            "end": args.session_end,
            "anchor": args.day_anchor,
        },
        "updated_at": f"{pd.Timestamp.now(tz=JST).isoformat()}",
    }
    out_json = Path(args.out_json)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # === テキスト ===
    sign = "▲" if pct_intraday >= 0 else "▼"
    out_text = Path(args.out_text)
    post = (
        f"{sign} {args.label} 日中スナップショット ({pd.Timestamp.now(tz=JST):%Y/%m/%d %H:%M})\n"
        f"{pct_intraday:+.2f}%（基準: {args.basis}）\n"
        f"#R_BANK9 #日本株\n"
    )
    out_text.write_text(post, encoding="utf-8")

    print("[make_intraday_post] snapshot saved  ->", out_png)
    print("[make_intraday_post] text saved      ->", out_text)
    print("[make_intraday_post] json saved      ->", out_json)


if __name__ == "__main__":
    main()
