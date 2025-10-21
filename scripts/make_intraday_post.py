#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Intraday snapshot generator (prev_close basis)
- CSVを読み、JSTでセッション時間に絞り込み
- 前日終値比（prev_close）で騰落率を算出（内部 = ratio、出力 = %）
- 画像/テキスト/JSONを出力
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from zoneinfo import ZoneInfo


# ====== Utils ================================================================

JST = ZoneInfo("Asia/Tokyo")


def to_jst_index(
    df: pd.DataFrame,
    dt_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    DatetimeIndexをJSTで返す。
    - dt_colがNoneなら、"Datetime" → "date" → "time" → 最初の列 など順に探索
    - すでにtz-awareならtz_convert、naiveならtz_localize
    """
    work = df.copy()

    if dt_col is None:
        cands = ["Datetime", "timestamp", "date", "time"]
        dt_col = next((c for c in cands if c in work.columns), None)
        if dt_col is None:
            # Unnamed: 0 のようなケース
            dt_col = work.columns[0]

    if dt_col not in work.columns:
        raise ValueError(f"CSVに対象列 '{dt_col}' がありません。候補: {list(work.columns)}")

    s = pd.to_datetime(work[dt_col], errors="coerce")
    if s.isna().all():
        raise ValueError(f"日時列 '{dt_col}' をdatetimeに変換できません。")

    # tzの付与/変換
    if s.dt.tz is None:
        s = s.dt.tz_localize(JST)
    else:
        s = s.dt.tz_convert(JST)

    work = work.drop(columns=[dt_col])
    work.index = s
    work = work.sort_index()
    return work


def between_session(df_jst: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """
    pandas.between_time を安全に呼ぶ（pandasのバージョン差異に配慮）
    """
    try:
        out = df_jst.between_time(start_time=start, end_time=end, inclusive="both")
    except TypeError:
        out = df_jst.between_time(start_time=start, end_time=end)
    return out


def ensure_ratio(series: pd.Series) -> pd.Series:
    """
    seriesが既に%であれば100で割ってratioに矯正する保険。
    しきい値は 20% を超える値が自然に出ることは稀、という前提。
    """
    s = series.astype(float).copy()
    if np.nanmax(np.abs(s.values)) > 20:  # 例: 45 などなら % と判定
        s = s / 100.0
    return s


def normalize_key(s: str) -> str:
    return s.lower().replace("_", "").replace("-", "").replace(" ", "")


def find_value_col(df: pd.DataFrame, key: str) -> str:
    """
    index_keyから最も尤もらしい列名を推定（大文字小文字・_/- の違いを吸収）。
    見つからなければ最後の数値列を返す。
    """
    nk = normalize_key(key)
    cols = list(df.columns)
    # 完全一致（正規化後）
    for c in cols:
        if normalize_key(c) == nk:
            return c
    # 部分一致
    for c in cols:
        if nk in normalize_key(c):
            return c
    # 数値列の最後をfallback
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if num_cols:
        return num_cols[-1]
    # どうにもならない
    return cols[-1]


def write_json(out_path: Path, payload: dict):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(out_path: Path, text: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")


# ====== Plot ================================================================

def plot_intraday(series_pct: pd.Series, out_png: Path, label: str, title: str):
    """
    画像の白余白をなくし、黒背景で保存
    series_pct: %（ratio*100）で渡す
    """
    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    # 図・軸の背景を黒で統一
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    ax.plot(series_pct.index, series_pct.values, linewidth=2.4)
    ax.set_title(title, color="white", fontsize=14, pad=10)
    ax.set_xlabel("Time", color="white")
    ax.set_ylabel("Change vs Prev Close (%)", color="white")

    ax.tick_params(axis="both", colors="white")
    for sp in ax.spines.values():
        sp.set_color("white")
    ax.grid(True, alpha=0.15, linewidth=0.8)

    # 余白ゼロ、facecolorも黒で保存
    plt.tight_layout(pad=0)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_png,
        bbox_inches="tight",
        pad_inches=0,
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )
    plt.close(fig)


# ====== Core ===============================================================

@dataclass
class Args:
    index_key: str
    csv: Path
    out_json: Path
    out_text: Path
    snapshot_png: Path
    session_start: str
    session_end: str
    day_anchor: str
    basis: str  # "prev_close" を想定
    value_type: str  # "ratio" を想定（内部表現）
    dt_col: Optional[str]
    label: str


def parse_args() -> Args:
    p = argparse.ArgumentParser(
        prog="make_intraday_post.py",
        description="Generate intraday snapshot & post (prev_close basis)",
    )
    p.add_argument("--index-key", required=True, dest="index_key")
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--out-json", required=True, type=Path, dest="out_json")
    p.add_argument("--out-text", required=True, type=Path, dest="out_text")
    p.add_argument("--snapshot-png", required=True, type=Path, dest="snapshot_png")
    p.add_argument("--session-start", required=True, dest="session_start")  # "09:00"
    p.add_argument("--session-end", required=True, dest="session_end")      # "15:30"
    p.add_argument("--day-anchor", required=True, dest="day_anchor")        # "09:00"
    p.add_argument("--basis", default="prev_close")
    p.add_argument("--value-type", default="ratio")  # ratio 固定推奨
    p.add_argument("--dt-col", default=None, dest="dt_col")
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
        label=a.label or a.index_key,
    )


def main():
    args = parse_args()

    # 1) CSV読込 → JST index
    raw = pd.read_csv(args.csv)
    df_jst = to_jst_index(raw, dt_col=args.dt_col)

    # 2) セッション時間で抽出
    sess = between_session(df_jst, args.session_start, args.session_end)
    if sess.empty:
        raise ValueError(
            f"セッション内データがありません。指定範囲: {args.session_start}–{args.session_end} JST / "
            f"データ範囲: {df_jst.index.min()} – {df_jst.index.max()}"
        )

    # 3) 値列の確定
    val_col = find_value_col(sess, args.index_key)
    series_raw = pd.to_numeric(sess[val_col], errors="coerce").dropna()
    if series_raw.empty:
        raise ValueError(f"CSVに有効な数値がありません。列='{val_col}'")

    # 4) basisに基づく ratio 算出（内部）
    if args.basis.lower() != "prev_close":
        raise ValueError("現状 'basis' は 'prev_close' のみサポートしています。")

    prev_close = series_raw.iloc[0]
    ratio = series_raw / prev_close - 1.0

    # CSV側が%だった場合の“二重換算”保険（想定外の大きさなら100で割る）
    ratio = ensure_ratio(ratio)

    # 5) 出力用シリーズ（%）
    pct = ratio * 100.0

    # 6) タイトル/文面
    anchor_dt = series_raw.index[0]
    title_ts = series_raw.index[-1].astimezone(JST).strftime("%Y/%m/%d %H:%M")
    title = f"{args.label} Intraday Snapshot ({title_ts})"

    last_pct = float(pct.iloc[-1])
    sign = "▲" if last_pct >= 0 else "▼"
    pct_text = f"{last_pct:+.2f}%"
    basis_label = "prev_close"

    text_lines = [
        f"{sign} {args.label} 日中スナップショット（{title_ts}）",
        f"{pct_text}（基準: {basis_label}）",
        f"#{args.index_key.replace('_', '-') } #日本株",
    ]
    write_text(args.out_text, "\n".join(text_lines))

    # 7) JSON
    payload = {
        "index_key": args.index_key,
        "label": args.label,
        "pct_intraday": last_pct,  # すでに%（UI でそのまま表示可）
        "basis": basis_label,
        "session": {
            "start": args.session_start,
            "end": args.session_end,
            "anchor": args.day_anchor,
        },
        "updated_at": pd.Timestamp.now(tz=JST).isoformat(),
    }
    write_json(args.out_json, payload)

    # 8) 画像
    plot_intraday(pct, args.snapshot_png, args.label, title)


if __name__ == "__main__":
    main()
