#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
R-BANK9 / ASTRA4 用：日中スナップショット作成スクリプト（修正版）
- CSVの日時列を堅牢にJSTのDatetimeIndexへ正規化
- basis: prev_close / open などに対応
- value_type: ratio(=小数) | percent(=100倍表示) を明示
- 余白の少ないチャート出力
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import json  # ← 追加：標準ライブラリで JSON を書く

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ====== ユーティリティ ======

JST = "Asia/Tokyo"


def to_jst_index(df: pd.DataFrame, dt_col: str) -> pd.DataFrame:
    """
    df[dt_col] を DatetimeIndex(JST) にして返す。
    - 文字列/オブジェクト → to_datetime
    - tz-aware → tz_convert(JST)
    - tz-naive  → tz_localize(JST)
    """
    if dt_col not in df.columns:
        raise ValueError(f"CSVに対象列 '{dt_col}' がありません。列={list(df.columns)}")

    s = df[dt_col]

    # まず Series を datetime へ
    dt = pd.to_datetime(s, errors="coerce")  # infer_datetime_format はデフォルト挙動に統合

    if dt.isna().all():
        raise ValueError("日時列の変換に失敗しました（すべてNaT）。原データ形式を確認してください。")

    # tz-aware かどうかを安全に判定
    is_tz_aware = getattr(dt.dtype, "tz", None) is not None

    if is_tz_aware:
        # 既に tz 付き → JST へ変換
        dt = dt.dt.tz_convert(JST)
    else:
        # tz なし → JST とみなしてローカライズ
        dt = dt.dt.tz_localize(JST)

    out = df.copy()
    out.index = dt
    out.drop(columns=[dt_col], inplace=True, errors="ignore")
    out = out[~out.index.duplicated(keep="last")]
    out.sort_index(inplace=True)
    return out


def pick_series(df: pd.DataFrame, index_key: str) -> pd.Series:
    """
    index_key（例: 'R_BANK9' / 'ASTRA4'）に対応する値列を推定して返す。
    想定列: そのまま, 大文字小文字差, *_mean など
    """
    candidates = [
        index_key,
        index_key.upper(),
        index_key.lower(),
        index_key.replace("-", "_"),
        index_key.replace("-", "_").upper(),
        f"{index_key}_mean",
        f"{index_key.upper()}_mean",
    ]
    for c in candidates:
        if c in df.columns:
            return df[c].astype("float64")

    # 最後の手段：数値列が1本だけならそれを採用
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) == 1:
        return df[num_cols[0]].astype("float64")

    raise ValueError(f"CSVに対象列が見つかりません。index_key='{index_key}', 候補={candidates}, 数値列={num_cols}")


def session_filter(
    s: pd.Series, start_hm: str, end_hm: str
) -> pd.Series:
    """
    JSTの時刻文字列 'HH:MM' で日中帯を抽出。
    """
    st_h, st_m = map(int, start_hm.split(":"))
    ed_h, ed_m = map(int, end_hm.split(":"))
    t0 = s.index.tz_convert(JST)
    mask = (
        (t0.hour > st_h) | ((t0.hour == st_h) & (t0.minute >= st_m))
    ) & (
        (t0.hour < ed_h) | ((t0.hour == ed_h) & (t0.minute <= ed_m))
    )
    return s[mask]


def compute_change(
    s: pd.Series, basis: str = "prev_close", value_type: str = "ratio"
) -> Tuple[pd.Series, float]:
    """
    指定basisに対する変化率（ratio or percent）を返す。
    - ratio   … 小数（0.0123 = +1.23%）
    - percent … 100倍（1.23 = +1.23%）
    戻り値: (series, 現在値[%表記])
    """
    if s.empty:
        raise ValueError("セッション内データが空です。")

    if basis == "prev_close":
        ref = s.iloc[0]
    elif basis == "open":
        ref = s.iloc[0]
    else:
        raise ValueError(f"未対応のbasisです: {basis}")

    # 変化率（小数）
    ratio = (s / ref) - 1.0

    if value_type == "percent":
        series = ratio * 100.0
        last_pct = float(series.iloc[-1])
    elif value_type == "ratio":
        series = ratio
        last_pct = float(ratio.iloc[-1] * 100.0)
    else:
        raise ValueError(f"未対応のvalue_typeです: {value_type}")

    return series, last_pct


def save_json(out_path: Path, payload: dict):
    """標準ライブラリ json を使用して書き出し（pandas.io.json は使用しない）"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_post_text(out_path: Path, label: str, last_pct: float, basis: str, now_jst: pd.Timestamp):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sign = "▲" if last_pct >= 0 else "▼"
    txt = f"{sign} {label} 日中スナップショット（{now_jst.strftime('%Y/%m/%d %H:%M')}）\n" \
          f"{last_pct:+.2f}%（基準: {basis}）\n" \
          f"#{label.replace('-', '').upper()} #日本株\n"
    out_path.write_text(txt, encoding="utf-8")


def save_plot(
    out_path: Path, series: pd.Series, title: str, label: str
):
    plt.figure(figsize=(11.5, 6.2), dpi=120)
    ax = plt.gca()
    ax.plot(series.index.tz_convert(JST), series.values, linewidth=2, label=label)

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Change vs Prev Close (%)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")

    # 余白削減
    plt.tight_layout(pad=0.6)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close()


# ====== メイン ======

@dataclass
class Args:
    index_key: str
    csv: str
    out_json: str
    out_text: str
    snapshot_png: str
    session_start: str
    session_end: str
    day_anchor: str
    basis: str
    value_type: str
    dt_col: str
    label: str


def parse_args() -> Args:
    p = argparse.ArgumentParser()
    p.add_argument("--index-key", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-text", required=True)
    p.add_argument("--snapshot-png", required=True)
    p.add_argument("--session-start", required=True)
    p.add_argument("--session-end", required=True)
    p.add_argument("--day-anchor", required=True)
    p.add_argument("--basis", default="prev_close")
    p.add_argument("--value-type", default="ratio", choices=["ratio", "percent"])
    p.add_argument("--dt-col", default="Datetime")
    p.add_argument("--label", default=None)
    a = p.parse_args()
    return Args(
        index_key=a.index_key,
        csv=a.csv,
        out_json=a.out_json,
        out_text=a.out_text,
        snapshot_png=a.snapshot_png,
        session_start=a.session_start,
        session_end=a.session_end,
        day_anchor=a.day_anchor,
        basis=a.basis,
        value_type=a.value_type,
        dt_col=a.dt_col,
        label=a.label or a.index_key.replace("_", "-")
    )


def main():
    args = parse_args()

    raw = pd.read_csv(args.csv)
    df = to_jst_index(raw, args.dt_col)

    series_raw = pick_series(df, args.index_key)

    # 日中帯抽出（JST）
    series = session_filter(series_raw, args.session_start, args.session_end)
    if series.empty:
        raise ValueError(
            f"セッション内データがありません。指定範囲: {args.session_start}–{args.session_end} JST / "
            f"データ範囲: {df.index.min()} – {df.index.max()}"
        )

    # 変化率に変換
    change_series, last_pct = compute_change(series, basis=args.basis, value_type=args.value_type)

    # 出力物
    now_jst = pd.Timestamp.now(tz=JST)
    payload = {
        "index_key": args.index_key,
        "label": args.label,
        "pct_intraday": float(last_pct),
        "basis": args.basis,
        "session": {"start": args.session_start, "end": args.session_end, "anchor": args.day_anchor},
        "updated_at": now_jst.isoformat(),
    }

    save_json(Path(args.out_json), payload)
    save_post_text(Path(args.out_text), args.label, last_pct, args.basis, now_jst)

    # 画像は%スケールで描画
    y = change_series if args.value_type == "percent" else change_series * 100.0
    title = f"{args.label} Intraday Snapshot ({now_jst.strftime('%Y/%m/%d %H:%M')})"
    save_plot(Path(args.snapshot_png), y, title=title, label=args.label)


if __name__ == "__main__":
    main()
