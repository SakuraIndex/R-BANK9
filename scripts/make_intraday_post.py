#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_intraday_post.py  — intraday スナップショット画像 / テキスト / JSON を生成

主な修正点（2025-10-21）:
- 価値の種類 (value_type) を堅牢に処理:
    ratio:  (x - 1) * 100  → % へ
    percent: そのまま % として使用
    price:   (x / x_at_09:00 - 1) * 100  → %（basis=open@09:00 のみ想定）
  加えてヒューリスティックでスケールを自動判定し、二重換算による
  「+168%」のような異常値を抑止。
- `basis=prev_close` のときは「最新値そのもの」を採用（既に前日終値比が入っている
  ラインアップに対応）。`basis=open@09:00` のときは (last - first) を採用。
- 画像まわりを全面調整: 背景/スパイン/マージンを最小化。白い枠線が出ないように。
"""

from __future__ import annotations
import argparse
from pathlib import Path
import json
import sys
from typing import Tuple

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ====== helpers ======

JST = "Asia/Tokyo"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--index-key", required=True)
    p.add_argument("--label", default=None)

    p.add_argument("--csv", required=True)
    p.add_argument("--dt-col", default="Datetime", help="Datetime column name in CSV")

    p.add_argument("--out-json", required=True)
    p.add_argument("--out-text", required=True)
    p.add_argument("--snapshot-png", required=True)

    p.add_argument("--session-start", default="09:00")
    p.add_argument("--session-end", default="15:30")
    p.add_argument("--day-anchor", default="09:00")

    p.add_argument("--basis", default="prev_close", choices=["prev_close", "open@09:00"])
    p.add_argument(
        "--value-type",
        default="ratio",
        choices=["ratio", "percent", "price", "auto"],
        help="ratio | percent | price | auto (推定)"
    )
    return p.parse_args()


def to_jst_index(df: pd.DataFrame, dt_col: str) -> pd.DataFrame:
    if dt_col not in df.columns:
        # よくあるケース: pandas の CSV 保存で Datetime が "Unnamed: 0" になる
        # -> その場合は index を流用
        if dt_col == "Datetime" and "Unnamed: 0" in df.columns:
            dt_col = "Unnamed: 0"
        elif dt_col == "Datetime" and df.index.name is not None:
            df = df.reset_index()

    if dt_col not in df.columns:
        raise ValueError(f"CSVに対象列 '{dt_col}' がありません。列={list(df.columns)}")

    ts = pd.to_datetime(df[dt_col], errors="coerce", utc=True)
    if ts.isna().all():
        # もし tz-aware なら tz_convert、tz-naive なら tz_localize
        ts = pd.to_datetime(df[dt_col], errors="coerce")
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize(JST, nonexistent="shift_forward", ambiguous="NaT")
        else:
            ts = ts.dt.tz_convert(JST)
    else:
        ts = ts.dt.tz_convert(JST)

    df = df.copy()
    df.index = ts
    return df.drop(columns=[dt_col])


def cut_session(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    # 当日 00:00 を基準にトリム
    if df.index.tz is None:
        df = df.tz_localize(JST)
    day = df.index[0].date()
    s = pd.to_datetime(f"{day} {start}").tz_localize(JST)
    e = pd.to_datetime(f"{day} {end}").tz_localize(JST)
    out = df.loc[(df.index >= s) & (df.index <= e)].copy()
    if out.empty:
        raise ValueError(
            f"セッション内データがありません。指定範囲: {start}–{end} JST / "
            f"データ範囲: {df.index.min()} – {df.index.max()}"
        )
    return out


def pick_series(df: pd.DataFrame, key_hint: str) -> Tuple[str, pd.Series]:
    """
    index_key（例: 'R_BANK9', 'ASTRA4'）に最も合致しそうな列を選ぶ。
    完全一致→部分一致→最後の手段で最初の数値列。
    """
    cols = list(df.columns)
    # 数値列のみ
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        raise ValueError("数値列が見つかりません。")

    # 完全一致
    for c in num_cols:
        if c == key_hint:
            return c, df[c]

    # 大文字小文字無視の部分一致（_mean や _ratio なども拾える）
    key_low = key_hint.lower()
    for c in num_cols:
        if key_low in c.lower():
            return c, df[c]

    # fallback
    return num_cols[0], df[num_cols[0]]


def detect_scale(series: pd.Series) -> str:
    """
    値のスケールを自動推定:
      - 中央値 ~1 を中心に 0.8–1.2 → ratio（基準=1）
      - 絶対値のメディアンが 0.5–10 → percent（既に%）
      - それ以外で典型的価格帯 (> 10) → price
    """
    s = series.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return "percent"
    med = float(np.median(s))
    if 0.8 <= med <= 1.2:
        return "ratio"
    if 0.5 <= abs(med) <= 10:
        return "percent"
    if abs(med) > 10:
        return "price"
    # デフォルト
    return "percent"


def to_percent_series(raw: pd.Series, value_type: str, basis: str, anchor_time: str) -> Tuple[pd.Series, str]:
    """
    任意スケールの series を「%」スケールの系列に変換して返す。
    basis は情報として返す（price のときは open@09:00 を強制）。
    """
    vt = value_type
    if vt == "auto":
        vt = detect_scale(raw)

    s = raw.copy()

    if vt == "ratio":
        # 1.0 が基準。→ (x - 1)*100
        percent = (s - 1.0) * 100.0
        return percent, basis

    if vt == "percent":
        # 既に % スケール（例: +3.5 → +3.5%）
        return s, basis

    if vt == "price":
        # 価格は当日 09:00 をアンカーとして % へ
        day = s.index[0].date()
        anchor_ts = pd.to_datetime(f"{day} {anchor_time}").tz_localize(JST)
        # 09:00 直近の値
        s_at = s.loc[:anchor_ts].iloc[-1] if not s.loc[:anchor_ts].empty else s.iloc[0]
        percent = (s / float(s_at) - 1.0) * 100.0
        return percent, "open@09:00"

    # 未知（保険）
    return s, basis


def latest_intraday_change(percent_series: pd.Series, basis: str) -> float:
    """
    basis が prev_close: 既に「前日終値比」の系列 → 最新値がそのまま答え
    basis が open@09:00: 09:00 基準 → last - first
    """
    if basis == "prev_close":
        return float(percent_series.iloc[-1])
    # open@09:00
    return float(percent_series.iloc[-1] - percent_series.iloc[0])


def plot_series_png(
    ts: pd.Series,
    label: str,
    title: str,
    out_path: Path,
) -> None:
    plt.style.use("dark_background")

    # 余白/枠線なしで描画
    fig = plt.figure(figsize=(11.5, 6.2), dpi=140, facecolor="#0b0f14")
    ax = fig.add_axes([0, 0, 1, 1])  # 余白ゼロ
    ax.set_facecolor("#0b0f14")

    # スパインを消す（白い枠線を除去）
    for sp in ax.spines.values():
        sp.set_visible(False)

    # ライン
    ax.plot(ts.index, ts.values, linewidth=2.2)

    # 軸/目盛りの色・グリッド微弱
    ax.tick_params(colors="#c7d1d9", labelsize=10)
    ax.grid(alpha=0.15)

    ax.set_title(title, color="#dfe7ee", fontsize=16, pad=10)
    ax.set_ylabel("Change vs Prev Close (%)", color="#aeb8c2", fontsize=11)
    ax.set_xlabel("Time", color="#aeb8c2", fontsize=11)
    ax.legend([label], loc="upper left", frameon=False)

    # 保存（完全に余白カット）
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def write_json(path: Path, index_key: str, label: str, pct_now: float,
               basis: str, session_start: str, session_end: str, anchor: str) -> None:
    payload = {
        "index_key": index_key,
        "label": label,
        "pct_intraday": pct_now,
        "basis": basis,
        "session": {
            "start": session_start,
            "end": session_end,
            "anchor": anchor,
        },
        "updated_at": pd.Timestamp.now(tz=JST).isoformat(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()

    # 入力
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    raw = pd.read_csv(csv_path)
    df = to_jst_index(raw, args.dt_col)

    # 対象列を選定
    col_name, series = pick_series(df, args.index_key)

    # セッション切り出し（先に series へ適用）
    df_sess = cut_session(series.to_frame(col_name), args.session_start, args.session_end)
    series_sess: pd.Series = df_sess[col_name]

    # % 系列へ正規化
    percent_series, effective_basis = to_percent_series(
        series_sess, args.value_type, args.basis, args.day_anchor
    )

    # 数値がおかしい時のガード（二重換算を簡易検知）
    # 例: 150%以上や -50%未満が連発する場合は ratio を percent に二重適用してる事が多い
    p99 = np.nanpercentile(percent_series.values, 99)
    if p99 > 50 and args.value_type in ("auto", "percent"):
        # 値が大きすぎれば、もともと ratio なのに % と解釈した可能性
        # → ratio 解釈に切り替え
        percent_series, effective_basis = to_percent_series(series_sess, "ratio", args.basis, args.day_anchor)

    # 現在の騰落率
    pct_now = latest_intraday_change(percent_series, effective_basis)

    # 出力（PNG / TEXT / JSON）
    label = args.label or args.index_key.replace("_", "-")
    title = f"{label} Intraday Snapshot ({pd.Timestamp.now(tz=JST):%Y/%m/%d %H:%M})"

    plot_series_png(percent_series, label, title, Path(args.snapshot_png))

    # テキスト
    Path(args.out_text).write_text(
        f"▲ {label} 日中スナップショット（{pd.Timestamp.now(tz=JST):%Y/%m/%d %H:%M}）\n"
        f"{pct_now:+.2f}%（基準: {effective_basis}）\n"
        f"#{args.index_key} #日本株\n",
        encoding="utf-8"
    )

    # JSON
    write_json(
        Path(args.out_json),
        args.index_key,
        label,
        pct_now,
        effective_basis,
        args.session_start,
        args.session_end,
        args.day_anchor,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
