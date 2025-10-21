#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
R-BANK9 intraday: make snapshot chart + post text + stats json

更新点
- --csv-unit {auto,percent,ratio} をサポート（既定: auto）
  * percent: CSVの各列は「前日比[%]」値（例 0.36 = +0.36%）
  * ratio  : CSVの各列は「前日比の小数倍率」（例 0.0036 = +0.36%）
  * auto   : ヒューリスティックで自動判定（0.95分位の絶対値 < 0.5 を ratio とみなす）
- タイムゾーン変換を安定化（naiveはUTCとみなしJSTへ）
- セッションは「最新行のJST日付」を基準に 09:00–15:30 等で抽出
- グラフはダーク背景、凡例（白枠）なし
- pct_intraday は「%値」をそのまま JSON に出力（例: +0.36% → 0.36）
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

JST = "Asia/Tokyo"


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
    basis: str
    value_type: str
    dt_col: str
    label: str
    csv_unit: str  # "auto" | "percent" | "ratio"


def parse_args() -> Args:
    p = argparse.ArgumentParser()
    p.add_argument("--index-key", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-text", required=True)
    p.add_argument("--snapshot-png", required=True)
    p.add_argument("--session-start", default="09:00")
    p.add_argument("--session-end", default="15:30")
    p.add_argument("--day-anchor", default="09:00")
    p.add_argument("--basis", default="prev_close")
    p.add_argument("--value-type", default="percent")  # 出力は%表記
    p.add_argument("--dt-col", default="Unnamed: 0")
    p.add_argument("--label", default="R-BANK9")
    p.add_argument(
        "--csv-unit",
        choices=["auto", "percent", "ratio"],
        default="auto",
        help="CSV の値の単位（前日比[%] or 小数倍率）。auto はヒューリスティックで判定。",
    )
    a = p.parse_args()
    return Args(
        index_key=a.index_key,
        csv=Path(a.csv),
        out_json=Path(a.out_json),
        out_text=Path(a.out_text),
        snapshot_png=Path(a.snapshot_png),
        session_start=a.session_start,
        session_end=a.session_end,
        day_anchor=a.day_anchor,
        basis=a.basis,
        value_type=a.value_type,
        dt_col=a.dt_col,
        label=a.label,
        csv_unit=a.csv_unit,
    )


# ---------- utils ----------

def _read_csv_jst(csv_path: Path, dt_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=[dt_col], index_col=dt_col)
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("Datetime index が取得できません。")
    # naive は UTC とみなし JST へ
    if idx.tz is None:
        idx = idx.tz_localize("UTC").tz_convert(JST)
    else:
        idx = idx.tz_convert(JST)
    df.index = idx
    # 数値化
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _session_slice(df_jst: pd.DataFrame, start_hm: str, end_hm: str) -> pd.DataFrame:
    if df_jst.empty:
        return df_jst
    last_day = df_jst.index[-1].date()  # JST date
    start = pd.Timestamp(f"{last_day} {start_hm}", tz=JST)
    end = pd.Timestamp(f"{last_day} {end_hm}", tz=JST)
    return df_jst.loc[(df_jst.index >= start) & (df_jst.index <= end)]


def _detect_unit_is_ratio(df: pd.DataFrame) -> bool:
    """
    値域から ratio かどうかを推定。
    0.95分位の絶対値が 0.5 未満の時は ratio とみなす（= 0.5% 未満が大勢）。
    """
    flat = pd.to_numeric(df.to_numpy().ravel(), errors="coerce")
    flat = flat[~np.isnan(flat)]
    if flat.size == 0:
        return False  # データなし → percent 扱い
    q95 = float(np.quantile(np.abs(flat), 0.95))
    return q95 < 0.5


def _to_index_percent(df_session: pd.DataFrame, csv_unit: str) -> pd.Series:
    """
    行方向に等加重平均し、%単位の Series を返す。
    csv_unit:
      - percent: そのまま平均
      - ratio  : 100倍して%に直して平均
      - auto   : 判定して上記いずれか
    """
    if df_session.empty:
        return pd.Series(dtype=float)

    unit = csv_unit
    if unit == "auto":
        unit = "ratio" if _detect_unit_is_ratio(df_session) else "percent"

    df_use = df_session.copy()
    if unit == "ratio":
        df_use = df_use * 100.0

    # 等加重平均（NaN除外）
    series_pct = df_use.mean(axis=1, skipna=True)

    # 極端なスパイクを軽く抑制（任意）
    series_pct = series_pct.clip(lower=-25.0, upper=25.0)
    return series_pct


def _format_sign_pct(x: float) -> str:
    return f"{x:+.2f}%"


def _save_text(path: Path, label: str, pct_last: float, basis: str, now_ts_jst: pd.Timestamp) -> None:
    lines = [
        f"{'▲' if pct_last >= 0 else '▼'} {label} 日中スナップショット ({now_ts_jst.strftime('%Y/%m/%d %H:%M')})",
        f"{_format_sign_pct(pct_last)}（基準: {basis}）",
        "#R_BANK9 #日本株",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _save_json(path: Path, index_key: str, label: str, pct_last: float,
               basis: str, start_hm: str, end_hm: str, anchor_hm: str,
               now_ts_jst: pd.Timestamp) -> None:
    import json
    payload = {
        "index_key": index_key,
        "label": label,
        "pct_intraday": float(pct_last),  # 例: 0.36 (= +0.36%)
        "basis": basis,
        "session": {"start": start_hm, "end": end_hm, "anchor": anchor_hm},
        "updated_at": now_ts_jst.isoformat(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _plot_series(path: Path, label: str, series_pct: pd.Series) -> None:
    # ダーク背景・凡例なし（白枠無し）
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    fig.patch.set_facecolor("#000000")
    ax.set_facecolor("#000000")

    if not series_pct.empty:
        ax.plot(series_pct.index, series_pct.values, linewidth=2.0)

        last_ts = pd.to_datetime(series_pct.index[-1]).tz_convert(JST)
        ax.set_title(f"{label} Intraday Snapshot ({last_ts.strftime('%Y/%m/%d %H:%M')})")
    else:
        ax.set_title(f"{label} Intraday Snapshot (no data)")

    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("#444444")
        ax.spines[side].set_linewidth(0.8)

    ax.set_ylabel("Change vs Prev Close (%)")
    ax.set_xlabel("Time")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)


# ---------- main ----------

def main() -> int:
    args = parse_args()
    print("=== Generate R-BANK9 intraday snapshot ===")
    print(f"VALUE_TYPE(default=percent) = {args.value_type}")
    print(f"CSV_UNIT = {args.csv_unit}")

    # 1) load & JST
    df = _read_csv_jst(args.csv, args.dt_col)
    print(f"[make_intraday_post] CSV: rows={len(df)}, cols={list(df.columns)}")

    # 2) session slice
    df_sess = _session_slice(df, args.session_start, args.session_end)
    if df_sess.empty:
        raise ValueError("セッション時間帯に該当するデータがありません。")

    # 3) % series
    series_pct = _to_index_percent(df_sess, args.csv_unit)

    # 4) last value (%)
    pct_last = float(np.round(series_pct.iloc[-1], 4)) if not series_pct.empty else 0.0
    now_ts_jst = pd.Timestamp.now(tz=JST)

    # 5) outputs
    _plot_series(Path(args.snapshot_png), args.label, series_pct)
    print(f"[make_intraday_post] snapshot saved -> {args.snapshot_png}")

    _save_text(Path(args.out_text), args.label, pct_last, args.basis, now_ts_jst)
    print(f"[make_intraday_post] text saved -> {args.out_text}")

    _save_json(Path(args.out_json), args.index_key, args.label, pct_last,
               args.basis, args.session_start, args.session_end, args.day_anchor, now_ts_jst)
    print(f"[make_intraday_post] json saved -> {args.out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
