#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

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
    csv_unit: str  # auto | percent | ratio


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
    p.add_argument("--value-type", default="percent")
    p.add_argument("--dt-col", default="Unnamed: 0")
    p.add_argument("--label", default="R-BANK9")
    p.add_argument(
        "--csv-unit",
        choices=["auto", "percent", "ratio"],
        default="auto",
        help="CSV の値の単位（percent=％値、ratio=小数倍率）。auto は内容から判定。",
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


# ---------- Datetime parsing (robust for JST-naive) ----------

def _parse_to_jst(series: pd.Series) -> Optional[pd.DatetimeIndex]:
    """Parse a datetime-like series and return tz-aware JST index.
    - If the parsed datetimes are tz-naive => treat as JST and tz_localize(JST)
    - If tz-aware => tz_convert(JST)
    Return None if parsing fails (all NaT).
    """
    ts = pd.to_datetime(series, errors="coerce", utc=False)
    if ts.isna().all():
        return None
    if getattr(ts.dt, "tz", None) is None:
        # Naive → すでに JST とみなしてローカライズ
        ts = ts.dt.tz_localize(JST, nonexistent="shift_forward", ambiguous="NaT")
    else:
        # tz あり（UTC 等）→ JST へ変換
        ts = ts.dt.tz_convert(JST)
    return pd.DatetimeIndex(ts)


def _try_parse_dt_col(df: pd.DataFrame, col: str) -> Optional[pd.DatetimeIndex]:
    if col in df.columns:
        return _parse_to_jst(df[col])
    return None


def _auto_find_datetime_index(df: pd.DataFrame) -> Tuple[pd.DatetimeIndex, str]:
    # 1) 先頭列を優先
    first = df.columns[0]
    ts = _parse_to_jst(df[first])
    if ts is not None and ts.notna().any():
        return ts, first
    # 2) 全列走査
    for c in df.columns:
        ts = _parse_to_jst(df[c])
        if ts is not None and ts.notna().any():
            return ts, c
    raise TypeError("Datetime index が取得できません。CSV の先頭行/列をご確認ください。")


def _read_csv_jst(csv_path: Path, prefer_col: Optional[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("CSV が空です。")

    # Datetime index 化（指定→自動検出）
    idx: Optional[pd.DatetimeIndex] = None
    used_col: Optional[str] = None
    if prefer_col and prefer_col in df.columns:
        idx = _try_parse_dt_col(df, prefer_col)
        used_col = prefer_col if idx is not None else None
    if idx is None:
        idx, used_col = _auto_find_datetime_index(df)

    df.index = idx
    if used_col in df.columns:
        df = df.drop(columns=[used_col])

    # 数値化
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 並び替え・重複除去
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    print(f"[read_csv] columns={list(df.columns)}  rows={len(df)}  used_dt_col={used_col}")
    print(f"[read_csv] time sample: {df.index.min()}  ..  {df.index.max()}")
    return df


# ---------- Session slicing & aggregation ----------

def _slice_window(df_jst: pd.DataFrame, day_ts: pd.Timestamp, start_hm: str, end_hm: str) -> pd.DataFrame:
    start = pd.Timestamp(f"{day_ts.date()} {start_hm}", tz=JST)
    end = pd.Timestamp(f"{day_ts.date()} {end_hm}", tz=JST)
    return df_jst.loc[(df_jst.index >= start) & (df_jst.index <= end)]


def _session_slice_with_fallbacks(df_jst: pd.DataFrame, start_hm: str, end_hm: str) -> pd.DataFrame:
    if df_jst.empty:
        return df_jst

    last_ts = df_jst.index[-1]
    # 1) 正規の 09:00–15:30（など）
    df = _slice_window(df_jst, last_ts, start_hm, end_hm)
    if not df.empty:
        return df

    # 2) 08:00–16:00 フォールバック（JST-naive→UTC 扱いだったデータ対策の緩衝）
    df = _slice_window(df_jst, last_ts, "08:00", "16:00")
    if not df.empty:
        return df

    # 3) 末尾から 120 本（最大 2 時間相当を想定）
    return df_jst.tail(120)


def _detect_unit_is_ratio(df: pd.DataFrame) -> bool:
    vals = pd.to_numeric(df.to_numpy().ravel(), errors="coerce")
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return False
    # だいたいのスケール判定：95%点が0.5未満 → ratio（= 0.5%未満が多い）
    return float(np.quantile(np.abs(vals), 0.95)) < 0.5


def _to_percent_series(df_session: pd.DataFrame, csv_unit: str) -> pd.Series:
    """
    df_session: セッション内の DataFrame
    優先度:
      1) 列名が index_key そのもの（例: 'R_BANK9' / 'R-BANK9'）ならその列を採用
      2) それが無ければ、数値列のみ等加重平均で算出（単位を ratio→percent に補正）
    返り値: ％（percent）単位の時系列
    """
    if df_session.empty:
        return pd.Series(dtype=float)

    cols: List[str] = [c for c in df_session.columns if pd.api.types.is_numeric_dtype(df_session[c])]
    if not cols:
        return pd.Series(dtype=float)

    dfn = df_session[cols].copy()

    keylike = [c for c in cols if c.strip().upper() in ("R_BANK9", "R-BANK9")]
    if keylike:
        s = pd.to_numeric(dfn[keylike[0]], errors="coerce")
        return s

    unit = csv_unit
    if unit == "auto":
        unit = "ratio" if _detect_unit_is_ratio(dfn) else "percent"

    if unit == "ratio":
        dfn = dfn * 100.0  # ratio→％

    s = dfn.mean(axis=1, skipna=True)
    s = s.clip(lower=-25.0, upper=25.0)
    return s


# ---------- Outputs ----------

def _format_sign_pct(x: float) -> str:
    return f"{x:+.2f}%"


def _save_text(path: Path, label: str, pct_last: float, basis: str, now_ts_jst: pd.Timestamp) -> None:
    lines = [
        f"{'▲' if pct_last >= 0 else '▼'} {label} 日中スナップショット ({now_ts_jst.strftime('%Y/%m/%d %H:%M')})",
        f"{_format_sign_pct(pct_last)}（基準: {basis}）",
        "#R_BANK9 #日本株",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _save_json(path: Path, index_key: str, label: str, pct_last_percent: float,
               basis: str, start_hm: str, end_hm: str, anchor_hm: str,
               now_ts_jst: pd.Timestamp) -> None:
    import json
    pct_ratio = float(np.round(pct_last_percent / 100.0, 6))  # ratio（小数）
    payload = {
        "index_key": index_key,
        "label": label,
        "pct_intraday": pct_ratio,  # 例: -0.0027 (= -0.27%)
        "basis": basis,
        "session": {"start": start_hm, "end": end_hm, "anchor": anchor_hm},
        "updated_at": now_ts_jst.isoformat(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _plot_series(path: Path, label: str, series_pct: pd.Series) -> None:
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    fig.patch.set_facecolor("#000000")
    ax.set_facecolor("#000000")

    if not series_pct.empty and series_pct.notna().any():
        series_pct = series_pct.dropna()
        last_val = float(series_pct.iloc[-1])
        line_color = "#00E5FF" if last_val >= 0 else "#FF4D4D"

        ax.axhline(0.0, color="#333333", linewidth=1.0, linestyle="-", zorder=0)
        ax.plot(series_pct.index, series_pct.values, color=line_color, linewidth=2.0, zorder=3)

        last_ts = pd.to_datetime(series_pct.index[-1]).tz_convert(JST)
        sign = "+" if last_val >= 0 else ""
        ax.set_title(
            f"{label} Intraday Snapshot ({last_ts.strftime('%Y/%m/%d %H:%M')})  {sign}{last_val:.2f}%",
            color="#D6E2EA"
        )
    else:
        ax.set_title(f"{label} Intraday Snapshot (no data)", color="#D6E2EA")

    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("#444444")
        ax.spines[side].set_linewidth(0.8)

    ax.set_ylabel("Change vs Prev Close (%)", color="#AAB8C2")
    ax.set_xlabel("Time", color="#AAB8C2")
    ax.tick_params(colors="#AAB8C2")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)


# ---------- main ----------

def main() -> int:
    args = parse_args()
    print("=== Generate R-BANK9 intraday snapshot ===")
    print(f"CSV_UNIT = {args.csv_unit}")

    # 1) CSV 読み込み（JST index）
    df = _read_csv_jst(args.csv, args.dt_col if args.dt_col else None)

    # 2) セッション抽出（フォールバック付き）
    df_sess = _session_slice_with_fallbacks(df, args.session_start, args.session_end)
    if df_sess.empty:
        print("[warn] セッションに該当するデータがありません。フォールバックも空でした。")

    # 3) ％系列に変換
    series_pct = _to_percent_series(df_sess, args.csv_unit)

    # 4) 終値（％）
    pct_last_percent = float(np.round(series_pct.dropna().iloc[-1], 4)) if not series_pct.empty and series_pct.notna().any() else 0.0
    now_ts_jst = pd.Timestamp.now(tz=JST)

    # 5) 出力
    _plot_series(Path(args.snapshot_png), args.label, series_pct)
    print(f"[make_intraday_post] snapshot saved -> {args.snapshot_png}")

    _save_text(Path(args.out_text), args.label, pct_last_percent, args.basis, now_ts_jst)
    print(f"[make_intraday_post] text saved -> {args.out_text}")

    _save_json(Path(args.out_json), args.index_key, args.label, pct_last_percent,
               args.basis, args.session_start, args.session_end, args.day_anchor, now_ts_jst)
    print(f"[make_intraday_post] json saved -> {args.out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
