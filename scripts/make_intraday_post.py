#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_intraday_post.py

任意のインデックス/合成シリーズの「日中スナップショット」を作成し、
- PNG（折れ線チャート）
- TXT（ポスト用の短文）
- JSON（メタ情報）

を出力します。

主な特徴
- CSV の日時列名や値列名を自動推定（--dt-col / --index-key で明示も可）
- タイムゾーンは自動判別して JST へ。tz-aware でも naive でも安全に処理
- セッション（start/end）は同一営業日内で抽出。データが無い場合は
  範囲と CSV の実データ範囲を丁寧に表示して失敗
- 基準値（basis）は列名で指定でき、無ければ prev_close 系の列を探索、
  それも無ければ「セッション最初値」を使用
- 目盛りの白枠（spines）を非表示にした黒背景のチャート
- 値の種類 value_type は受け取りますが、出力は “％表示” を標準化
  （実用上もっとも分かりやすいため）

使い方（例）
python scripts/make_intraday_post.py \
  --index-key "R_BANK9" \
  --csv docs/outputs/rbank9_intraday.csv \
  --out-json docs/outputs/rbank9_stats.json \
  --out-text docs/outputs/rbank9_post_intraday.txt \
  --snapshot-png docs/outputs/rbank9_intraday.png \
  --session-start "09:00" \
  --session-end   "15:30" \
  --day-anchor    "09:00" \
  --basis "prev_close" \
  --value-type "ratio" \
  --dt-col "Unnamed: 0" \
  --label "R-BANK9"
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


JST = "Asia/Tokyo"


# ---------- ユーティリティ ----------

def to_jst_index(df: pd.DataFrame, dt_col: str) -> pd.DataFrame:
    """任意の日時列を受け取り、JST の DatetimeIndex に変換して返す。
    tz-aware -> tz_convert(JST)
    naive    -> tz_localize(JST)
    """
    if dt_col not in df.columns:
        raise ValueError(f"CSV に日時列 '{dt_col}' が見つかりません。候補: {list(df.columns)}")

    ts = pd.to_datetime(df[dt_col], errors="coerce", utc=False)
    if ts.isna().all():
        raise ValueError(f"日時列 '{dt_col}' を datetime に変換できません。値の例: {df[dt_col].head(3).tolist()}")

    # 可能なら tz 情報を確認
    if ts.dt.tz is not None:
        # すでに tz-aware
        ts = ts.dt.tz_convert(JST)
    else:
        # naive → JST とみなす（データが JST で吐かれている前提が多いため）
        ts = ts.dt.tz_localize(JST)

    out = df.copy()
    out.index = ts
    return out.drop(columns=[dt_col])


def parse_hhmm_jst(s: str, anchor_date: pd.Timestamp) -> pd.Timestamp:
    m = re.fullmatch(r"(\d{2}):(\d{2})", s.strip())
    if not m:
        raise ValueError(f"時刻は HH:MM 形式で指定してください（例: 09:00）。受け取った値: {s}")
    hh, mm = int(m.group(1)), int(m.group(2))
    # anchor_date は JST の tz-aware を想定
    return anchor_date.replace(hour=hh, minute=mm, second=0, microsecond=0)


def filter_session(df_jst: pd.DataFrame, session_start: str, session_end: str, anchor_time: str) -> pd.DataFrame:
    """同一営業日（= データの最後のタイムスタンプの日）における
    セッション（start/end）だけを抽出する。
    """
    if df_jst.empty:
        raise ValueError("CSV が空です。")

    last_ts: pd.Timestamp = df_jst.index.max()
    if last_ts.tz is None:
        # 念のため
        last_ts = last_ts.tz_localize(JST)

    anchor_date = last_ts.tz_convert(JST).normalize()
    # 「ラベル用」の日付は末尾のデータ日付
    day_start = parse_hhmm_jst(session_start, anchor_date)
    day_end = parse_hhmm_jst(session_end, anchor_date)

    if day_end <= day_start:
        # 23:55〜00:05 のような跨ぎには対応しない方針
        raise ValueError(f"セッション時刻の順序が不正です（start={session_start}, end={session_end}）。同一営業日内で指定してください。")

    out = df_jst.loc[(df_jst.index >= day_start) & (df_jst.index <= day_end)].copy()

    if out.empty:
        rng = f"{day_start.strftime('%Y-%m-%d %H:%M %Z')} 〜 {day_end.strftime('%Y-%m-%d %H:%M %Z')}"
        have = f"{df_jst.index.min().strftime('%Y-%m-%d %H:%M %Z')} 〜 {df_jst.index.max().strftime('%Y-%m-%d %H:%M %Z')}"
        raise ValueError(f"セッション内データがありません。指定範囲: {session_start}–{session_end} JST / データ範囲: {have}（抽出対象日: {rng}）")

    return out


def pick_value_column(df: pd.DataFrame, index_key: str) -> str:
    """インデックス/系列の値列を推定する。
    - index_key（完全一致）
    - 大文字/小文字/ハイフン/アンダースコアの差を吸収した一致
    - "<index_key>_mean" などの部分一致
    - 最後の数値列（日時列や 'Unnamed: 0' を除く）
    """
    cols = list(df.columns)

    normalized = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())

    target_norm = normalized(index_key)
    # 1) 完全一致（ゆるい正規化後）
    for c in cols:
        if normalized(c) == target_norm:
            return c

    # 2) _mean, _index, - などを吸収
    for c in cols:
        if target_norm in normalized(c):
            return c

    # 3) 末尾の数値列を最後の手段として使う（Unnamed, index など除外）
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ValueError(f"数値列が見つかりません。columns={cols}")
    # Unnamed 系を除外
    numeric_cols = [c for c in numeric_cols if not re.match(r"unnamed", c, flags=re.I)]
    if not numeric_cols:
        raise ValueError(f"数値列が見つかりません（Unnamed を除外後）。columns={cols}")

    return numeric_cols[-1]


def find_basis_value(df: pd.DataFrame, val_col: str, basis_col: Optional[str]) -> float:
    """基準値の探索ロジック。
    優先順位：
      1) --basis で列名が指定されていればそれを最優先
      2) prev_close / close_yesterday 等の列があれば使用
      3) なければ「セッション最初値」
    """
    candidates: list[str] = []
    if basis_col:
        candidates.append(basis_col)

    # よくある列名
    for patt in ("prev_close", "previous_close", "close_yesterday", "prevclose", "base"):
        for c in df.columns:
            if re.search(patt, c, flags=re.I):
                candidates.append(c)

    for c in candidates:
        if c in df.columns:
            base = float(df[c].iloc[0])
            if not (math.isnan(base) or base == 0.0):
                return base

    # fallback: セッションの最初値
    base = float(df[val_col].iloc[0])
    if math.isnan(base) or base == 0.0:
        raise ValueError("基準値を決定できません（NaN または 0）。--basis で列名を指定するか、CSV を確認してください。")
    return base


def compute_change_percent(df: pd.DataFrame, val_col: str, basis_col: Optional[str]) -> pd.Series:
    """％変化（前日終値や最初値に対する変化）を返す。単位は percent（例: 3.5 は +3.5%）。"""
    base = find_basis_value(df, val_col, basis_col)
    series = pd.to_numeric(df[val_col], errors="coerce")
    if series.isna().all():
        raise ValueError(f"値列 '{val_col}' に数値がありません。")

    chg_ratio = series / float(base) - 1.0
    chg_pct = chg_ratio * 100.0
    return chg_pct


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
    basis: Optional[str]
    value_type: str  # 互換のため受け取るが、出力は percent に統一
    dt_col: Optional[str]
    label: Optional[str]


def parse_args() -> Args:
    p = argparse.ArgumentParser()
    p.add_argument("--index-key", required=True, dest="index_key")
    p.add_argument("--csv", required=True)
    p.add_argument("--out-json", required=True, dest="out_json")
    p.add_argument("--out-text", required=True, dest="out_text")
    p.add_argument("--snapshot-png", required=True, dest="snapshot_png")
    p.add_argument("--session-start", required=True, dest="session_start")
    p.add_argument("--session-end", required=True, dest="session_end")
    p.add_argument("--day-anchor", required=True, dest="day_anchor")
    p.add_argument("--basis", required=False, default=None, dest="basis",
                   help="基準列名（例: prev_close）。無い場合は prev_close 系を探索し、見つからなければ最初値。")
    p.add_argument("--value-type", required=False, default="percent",
                   help="互換用の引数。入出力とも最終的には％表示で統一します（ratio を渡しても％で出力）。")
    p.add_argument("--dt-col", required=False, default=None,
                   help="日時列名。省略時は 'Datetime' や先頭列などを自動推定。")
    p.add_argument("--label", required=False, default=None,
                   help="凡例/タイトル用ラベル。省略時は index_key を使用。")
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
        value_type=str(a.value_type or "percent").lower(),
        dt_col=a.dt_col,
        label=a.label,
    )


# ---------- 描画 ----------

def plot_snapshot(
    ts: pd.DatetimeIndex,
    y_pct: pd.Series,
    label: str,
    title_day: str,
    out_png: str,
) -> None:
    """黒背景・枠線ナシのスナップショットを描画して保存。"""
    plt.close("all")
    fig, ax = plt.subplots(figsize=(12, 6), facecolor="black")
    ax.set_facecolor("black")

    # 線色は視認性の高いシアン
    ax.plot(ts, y_pct.values, linewidth=2.0, color="#10e0e0", label=label)

    # 枠線（spines）を消す
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 目盛り色
    ax.tick_params(colors="#c8c8c8")
    ax.yaxis.label.set_color("#c8c8c8")
    ax.xaxis.label.set_color("#c8c8c8")

    ax.set_title(f"{label} Intraday Snapshot ({title_day})", color="#eaeaea", fontsize=14, pad=12)
    ax.set_xlabel("Time")
    ax.set_ylabel("Change vs Prev Close (%)")

    ax.legend(facecolor="black", edgecolor="none", labelcolor="#eaeaea")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------- 本体 ----------

def main() -> None:
    args = parse_args()

    # CSV 読み込み
    raw = pd.read_csv(args.csv)

    # 日時列の推定（指定 > 'Datetime' > 'date' > 先頭列）
    dt_col = args.dt_col
    if dt_col is None:
        for cand in ("Datetime", "datetime", "Date", "date", "timestamp", "Timestamp"):
            if cand in raw.columns:
                dt_col = cand
                break
        if dt_col is None:
            # 先頭列を日時列とみなす（Unnamed: 0 など想定）
            dt_col = raw.columns[0]

    df_jst = to_jst_index(raw, dt_col=dt_col)

    # 値列の特定
    val_col = pick_value_column(df_jst, args.index_key)

    # 同一営業日のセッション抽出
    df_sess = filter_session(df_jst, args.session_start, args.session_end, args.day_anchor)

    # ％変化を算出（出力は％に統一）
    y_pct = compute_change_percent(df_sess, val_col=val_col, basis_col=args.basis)

    # 見出し（タイトル用日付）
    last_ts = df_sess.index.max().tz_convert(JST)
    title_day = last_ts.strftime("%Y/%m/%d %H:%M")

    # チャート
    label = args.label or args.index_key
    plot_snapshot(df_sess.index, y_pct, label=label, title_day=title_day, out_png=args.snapshot_png)

    # ポスト用テキスト
    latest_pct = float(y_pct.iloc[-1])
    arrow = "▲" if latest_pct >= 0 else "▼"
    sign = "+" if latest_pct >= 0 else ""
    post_lines = [
        f"{arrow} {label} 日中スナップショット（{title_day}）",
        f"{sign}{latest_pct:.2f}%（基準: {args.basis or 'prev_close/first'}）",
        f"#{args.index_key.replace('_', '-')}"  # ハッシュタグ的に
    ]
    with open(args.out_text, "w", encoding="utf-8") as f:
        f.write("\n".join(post_lines))

    # JSON
    stats = {
        "index_key": args.index_key,
        "label": label,
        "pct_intraday": latest_pct,     # ％で保存（例: 3.52 は +3.52%）
        "basis": args.basis or "prev_close/first",
        "session": {
            "start": args.session_start,
            "end": args.session_end,
            "anchor": args.day_anchor,
        },
        "updated_at": last_ts.isoformat(),
    }
    import json
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
