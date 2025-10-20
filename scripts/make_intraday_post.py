#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_intraday_post.py

CSV を読み込み、日時列を JST に変換してセッション時間で抽出し、
スナップショット PNG・ポスト文（TXT）・統計 JSON を出力します。

主な強化点（堅牢化）:
- --dt-col auto で日時列を自動検出（候補: Datetime, datetime, date, time, timestamp, Timestamp, 日時, 日付, 時刻, Unnamed: 0 など）
- DataFrame の index が日時っぽい場合の自動利用
- 値列の自動選択（index_key/label に一致・_mean/_avg を優先・唯一の数値列の自動採用）
- JST 変換の安全化（UTC/naive 両対応）
- セッション抽出は開始・終了を含む（>= start & <= end）
- エラー時の列一覧を明示してデバッグ容易に
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter


# ---------------------------
# Utilities
# ---------------------------

JST = "Asia/Tokyo"


def _to_datetime_jst(series: pd.Series) -> pd.DatetimeIndex:
    """
    与えられた Series を日時として解釈し JST の DatetimeIndex を返す。
    - tz-aware(UTC) → JST に変換
    - naive        → まず naive→UTC と仮定して JST に変換（失敗時は naive→JST として扱う）
    """
    # まず厳密に
    try:
        dt = pd.to_datetime(series, utc=True, errors="raise")
        # 既に tz-aware ならそのまま JST へ
        if getattr(dt, "tz", None) is not None:
            return dt.tz_convert(JST)
    except Exception:
        # 次で緩めに解釈
        pass

    # 緩やかに（欠損は NaT）
    dt2 = pd.to_datetime(series, errors="coerce")

    # tz がない場合: UTC と仮定して JST へ
    if getattr(dt2, "tz", None) is None:
        try:
            dt2 = dt2.dt.tz_localize("UTC").tz_convert(JST)
        except Exception:
            # UTC 仮定で失敗するなら JST ローカライズを試す
            dt2 = dt2.dt.tz_localize(JST)

    else:
        dt2 = dt2.tz_convert(JST)

    return dt2


def to_jst_index(raw: pd.DataFrame, dt_col: Optional[str]) -> pd.DataFrame:
    """
    dt_col が 'auto' または None のときは日時列を自動検出。
    候補にヒットしなければ、index が日時っぽいかを試し、最後に明示的エラー。
    """
    if dt_col and dt_col.lower() != "auto":
        if dt_col not in raw.columns:
            raise ValueError(f"CSVに指定の日時列 '{dt_col}' が見つかりません。列={list(raw.columns)}")
        idx = _to_datetime_jst(raw[dt_col])
        out = raw.copy()
        out = out.set_index(idx)
        return out

    # 自動検出
    candidates = [
        "Datetime", "datetime", "date", "time", "timestamp", "Timestamp",
        "日時", "日付", "時刻", "Unnamed: 0"
    ]
    for c in candidates:
        if c in raw.columns:
            try:
                idx = _to_datetime_jst(raw[c])
                out = raw.copy()
                out = out.set_index(idx)
                return out
            except Exception:
                # 次の候補へ
                pass

    # index が日時に見えるか
    try:
        idx = _to_datetime_jst(pd.Series(raw.index))
        if idx.notna().any():
            out = raw.copy()
            out.index = idx
            return out
    except Exception:
        pass

    raise ValueError(
        f"CSVに日時列が見つかりません（auto）。"
        f"候補={candidates} / 列={list(raw.columns)}"
    )


def _numeric_columns(df: pd.DataFrame) -> List[str]:
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return num_cols


def pick_value_series(df: pd.DataFrame, index_key: str, label: Optional[str] = None) -> Tuple[pd.Series, str]:
    """
    値列を決定して返す。返り値は (Series, 列名)
    優先順位:
      1) index_key と完全一致（大文字小文字は無視）する列
      2) label と一致
      3) *_mean / *_avg を持つ数値列
      4) 数値列が一つだけならそれ
      5) だめならエラー（列名を表示）
    """
    cols_lower = {c.lower(): c for c in df.columns}
    if index_key:
        key = index_key.lower()
        if key in cols_lower:
            c = cols_lower[key]
            if pd.api.types.is_numeric_dtype(df[c]):
                return df[c].astype(float), c

    if label:
        lab = label.lower()
        if lab in cols_lower:
            c = cols_lower[lab]
            if pd.api.types.is_numeric_dtype(df[c]):
                return df[c].astype(float), c

    # *_mean / *_avg
    mean_like = [c for c in _numeric_columns(df) if c.lower().endswith(("_mean", "_avg"))]
    if mean_like:
        c = mean_like[0]
        return df[c].astype(float), c

    num_cols = _numeric_columns(df)
    if len(num_cols) == 1:
        return df[num_cols[0]].astype(float), num_cols[0]

    # ここまで来たら選べない
    raise ValueError(
        "値列を特定できませんでした。index_key/label と合致する数値列がなく、"
        "かつ *_mean/_avg も見つからず、唯一の数値列でもありません。\n"
        f"index_key={index_key}, label={label}, 数値列={num_cols}, 全列={list(df.columns)}"
    )


def parse_hhmm_to_jst_time(hhmm: str, anchor_date: pd.Timestamp) -> pd.Timestamp:
    """
    '09:00' などを同日の JST 時刻にして返す。
    `anchor_date` は JST の DatetimeIndex のいずれか（その日の基準日として使用）。
    """
    hh, mm = map(int, hhmm.split(":"))
    base = anchor_date.floor("D")  # その日の 00:00 JST
    return base + pd.Timedelta(hours=hh, minutes=mm)


def filter_session(df_jst: pd.DataFrame, start_hhmm: str, end_hhmm: str) -> pd.DataFrame:
    """
    start <= t <= end を満たす範囲で抽出（JST）。
    """
    if df_jst.empty:
        raise ValueError("データが空です。CSV を確認してください。")

    anchor_dt = df_jst.index[0]
    if getattr(anchor_dt, "tz", None) is None:
        # 念のため JST に
        anchor_dt = anchor_dt.tz_localize(JST)

    start = parse_hhmm_to_jst_time(start_hhmm, anchor_dt)
    end = parse_hhmm_to_jst_time(end_hhmm, anchor_dt)

    out = df_jst[(df_jst.index >= start) & (df_jst.index <= end)]
    if out.empty:
        raise ValueError(
            f"セッション内データがありません。指定範囲: {start_hhmm}–{end_hhmm} JST / "
            f"データ範囲: {df_jst.index.min()} – {df_jst.index.max()}"
        )
    return out


def compute_latest_pct(value: float, value_type: str) -> float:
    """
    最新値を % で返す。
      - value_type == 'ratio'   → value * 100 （例: 0.035 → 3.5）
      - value_type == 'percent' → そのまま（例: 3.5 → 3.5）
    """
    vt = (value_type or "ratio").lower()
    if vt == "ratio":
        return value * 100.0
    elif vt == "percent":
        return value
    else:
        raise ValueError(f"不正な value_type: {value_type}（ratio または percent）")


def format_sign_pct(pct: float) -> str:
    sign = "▲" if pct >= 0 else "▼"
    return f"{sign}{abs(pct):.2f}%"


def make_plot_png(
    ts: pd.Series,
    out_png: Path,
    title: str,
    ylabel: str,
    label: str
) -> None:
    """
    黒背景・シアンの線、余計な枠線なしで描画。
    """
    plt.close("all")
    fig, ax = plt.subplots(figsize=(12, 6), facecolor="#111111")
    fig.patch.set_alpha(1.0)
    ax.set_facecolor("#111111")

    ax.plot(ts.index, ts.values, linewidth=2.0, color="#00E5FF", label=label)

    ax.set_title(title, color="white", fontsize=14, pad=12)
    ax.set_ylabel(ylabel, color="white")
    ax.set_xlabel("Time", color="white")

    # 目盛り色
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    # x 軸フォーマット
    ax.xaxis.set_major_formatter(DateFormatter("%m-%d %H"))

    # グリッド
    ax.grid(True, color="#333333", linestyle="-", linewidth=0.6, alpha=0.6)

    # 枠線消し
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 凡例
    ax.legend(facecolor="#111111", edgecolor="none", labelcolor="white")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--index-key", required=True, help="列名の候補（優先的に使う値列のキー）")
    p.add_argument("--csv", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-text", required=True)
    p.add_argument("--snapshot-png", required=True)

    p.add_argument("--session-start", required=True, help="HH:MM JST")
    p.add_argument("--session-end", required=True, help="HH:MM JST")
    p.add_argument("--day-anchor", required=True, help="ラベル用: HH:MM JST")

    p.add_argument("--basis", required=True, help="基準ラベル（例: prev_close / open@09:00 など）")
    p.add_argument("--value-type", default="ratio", help="ratio | percent")
    p.add_argument("--dt-col", default="auto", help="日時列名。'auto' なら自動検出")
    p.add_argument("--label", default=None, help="チャート凡例・タイトル内ラベル（省略可）")

    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV が見つかりません: {csv_path}")

    raw = pd.read_csv(csv_path)

    # 1) JST index 化
    df_jst = to_jst_index(raw, args.dt_col)

    # 2) 値列の決定
    series, used_col = pick_value_series(df_jst, index_key=args.index_key, label=args.label or args.index_key)

    # 3) セッション抽出
    sess = filter_session(df_jst.assign(_val=series)["_val"], args.session_start, args.session_end)

    # 4) 最新値 → % に正規化
    latest_ratio_or_percent = sess.iloc[-1]
    latest_pct = compute_latest_pct(latest_ratio_or_percent, args.value_type)

    # 5) 出力（PNG / TXT / JSON）
    label = args.label or args.index_key
    title_ts = pd.Timestamp.now(tz=JST).strftime("%Y/%m/%d %H:%M")
    title = f"{label} Intraday Snapshot ({title_ts})"

    ylabel = "Change vs Prev Close (%)" if args.basis.lower().startswith("prev") else "Change vs Anchor (%)"

    make_plot_png(sess, Path(args.snapshot_png), title=title, ylabel=ylabel, label=label)

    # TXT
    sign_line = format_sign_pct(latest_pct)
    text_lines = [
        f"▲ {label} 日中スナップショット ({title_ts})" if latest_pct >= 0 else f"▼ {label} 日中スナップショット ({title_ts})",
        f"{sign_line} (基準: {args.basis})",
        f"#{label} #日本株",
    ]
    out_text = Path(args.out_text)
    out_text.parent.mkdir(parents=True, exist_ok=True)
    out_text.write_text("\n".join(text_lines), encoding="utf-8")

    # JSON
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "index_key": args.index_key,
        "label": label,
        "pct_intraday": float(latest_pct),
        "basis": args.basis,
        "session": {
            "start": args.session_start,
            "end": args.session_end,
            "anchor": args.day_anchor,
        },
        "updated_at": pd.Timestamp.now(tz=JST).isoformat(),
        "source": {
            "csv": str(csv_path),
            "dt_col_used": args.dt_col,
            "value_col_used": used_col,
            "value_type": args.value_type,
        },
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
