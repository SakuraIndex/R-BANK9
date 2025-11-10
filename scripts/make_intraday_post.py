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

# タイムゾーン：pandas は IANA 文字列でも扱える
JST = "Asia/Tokyo"


# =========================
# Args
# =========================
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
    # 既存CSVの日時列名が不定のため、指定があれば優先し、無ければ自動検出
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


# =========================
# CSV 読み込み（ロバスト）
# =========================
def _try_parse_dt_col(df: pd.DataFrame, col: str) -> Optional[pd.DatetimeIndex]:
    if col in df.columns:
        ts = pd.to_datetime(df[col], utc=True, errors="coerce")
        if ts.notna().any():
            return ts.dt.tz_convert(JST)
    return None


def _auto_find_datetime_index(df: pd.DataFrame) -> Tuple[pd.DatetimeIndex, str]:
    # 1) 先頭列を優先
    first = df.columns[0]
    ts = pd.to_datetime(df[first], utc=True, errors="coerce")
    if ts.notna().any():
        return ts.dt.tz_convert(JST), first
    # 2) 全列走査
    for c in df.columns:
        ts = pd.to_datetime(df[c], utc=True, errors="coerce")
        if ts.notna().any():
            return ts.dt.tz_convert(JST), c
    raise TypeError("Datetime index が取得できません。CSV の先頭行/列をご確認ください。")


def _read_csv_jst(csv_path: Path, prefer_col: Optional[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("CSV が空です。")

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
    return df


# =========================
# セッション抽出・整形
# =========================
def _session_slice(df_jst: pd.DataFrame, start_hm: str, end_hm: str) -> pd.DataFrame:
    if df_jst.empty:
        return df_jst
    last_day = df_jst.index[-1].date()
    start = pd.Timestamp(f"{last_day} {start_hm}", tz=JST)
    end = pd.Timestamp(f"{last_day} {end_hm}", tz=JST)
    return df_jst.loc[(df_jst.index >= start) & (df_jst.index <= end)]


def _build_grid_for_day(last_day: pd.Timestamp | pd.DatetimeIndex, start_hm: str, end_hm: str) -> pd.DatetimeIndex:
    if isinstance(last_day, pd.DatetimeIndex):
        base = last_day[-1]
    else:
        base = last_day
    sh, sm = map(int, start_hm.split(":"))
    eh, em = map(int, end_hm.split(":"))
    s = base.tz_convert(JST).replace(hour=sh, minute=sm, second=0, microsecond=0)
    e = base.tz_convert(JST).replace(hour=eh, minute=em, second=0, microsecond=0)
    return pd.date_range(s, e, freq="5T", tz=JST)


def _detect_unit_is_ratio(df: pd.DataFrame) -> bool:
    vals = pd.to_numeric(df.to_numpy().ravel(), errors="coerce")
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return False
    # スケール判定：95%点が0.5未満 → ratio とみなす
    return float(np.quantile(np.abs(vals), 0.95)) < 0.5


def _pick_component_cols(df: pd.DataFrame) -> List[str]:
    """指数列や非数値列を除外して、構成銘柄候補を抽出"""
    ban = {"R_BANK9", "R-BANK9", "INDEX", "Index", "index"}
    cols: List[str] = []
    for c in df.columns:
        if c in ban:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() > 0:
            cols.append(c)
    return cols


def _align_and_filter(df_sess: pd.DataFrame, start_hm: str, end_hm: str,
                      comp_cols: List[str], min_ratio: float = 1.0) -> pd.DataFrame:
    """5分グリッドに再インデックス(ffill)し、十分に揃っていない行をドロップ"""
    grid = _build_grid_for_day(df_sess.index, start_hm, end_hm)
    aligned = df_sess.reindex(grid).ffill()

    if not comp_cols:
        return aligned

    need = int(len(comp_cols) * min_ratio + 1e-9)
    mask = aligned[comp_cols].count(axis=1) >= need
    aligned = aligned.loc[mask]

    # すべて落ちた場合は、閾値を少し緩めて救済
    if aligned.empty and min_ratio > 0.8 and len(comp_cols) >= 2:
        need = max(2, int(len(comp_cols) * 0.8 + 1e-9))
        mask = df_sess.reindex(grid).ffill()[comp_cols].count(axis=1) >= need
        aligned = df_sess.reindex(grid).ffill().loc[mask]

    return aligned


def _compute_index_percent(df_aligned: pd.DataFrame, comp_cols: List[str], csv_unit: str) -> pd.Series:
    """構成銘柄の等ウェイト平均から R_BANK9 を再計算（％）"""
    if csv_unit == "auto":
        unit = "ratio" if _detect_unit_is_ratio(df_aligned[comp_cols]) else "percent"
    else:
        unit = csv_unit

    dfn = df_aligned[comp_cols].astype(float)
    if unit == "ratio":
        dfn = dfn * 100.0  # 倍率→％

    s = dfn.mean(axis=1, skipna=True)
    # 壊れ値対策で軽くクリップ（視覚崩れ防止）
    s = s.clip(lower=-25.0, upper=25.0)
    s.name = "R_BANK9"
    return s


# =========================
# 出力
# =========================
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
    pct_ratio = float(np.round(pct_last_percent / 100.0, 6))  # ％→ratio
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
    """黒背景・最終値で色分け"""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    fig.patch.set_facecolor("#000000")
    ax.set_facecolor("#000000")

    if not series_pct.empty:
        last_val = float(series_pct.iloc[-1])
        line_color = "#00E5FF" if last_val >= 0 else "#FF4D4D"

        ax.axhline(0.0, color="#333333", linewidth=1.0, linestyle="-", zorder=0)
        ax.plot(series_pct.index, series_pct.values, color=line_color, linewidth=2.2, zorder=3)

        last_ts = pd.to_datetime(series_pct.index[-1]).tz_convert(JST)
        sign = "+" if last_val >= 0 else ""
        ax.set_title(f"{label} Intraday Snapshot ({last_ts.strftime('%Y/%m/%d %H:%M')})  {sign}{last_val:.2f}%",
                     color="#D6E2EA")
    else:
        ax.set_title(f"{label} Intraday Snapshot (no data)", color="#D6E2EA")

    # 枠線・軸
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


# =========================
# main
# =========================
def main() -> int:
    args = parse_args()
    print("=== Generate R-BANK9 intraday snapshot ===")
    print(f"CSV_UNIT = {args.csv_unit}")

    # 1) CSV 読み込み（JST index）
    df = _read_csv_jst(args.csv, args.dt_col if args.dt_col else None)

    # 2) セッション抽出
    df_sess = _session_slice(df, args.session_start, args.session_end)
    if df_sess.empty:
        raise ValueError("セッション時間帯に該当するデータがありません。")

    # 3) 構成銘柄列の抽出（指数列は使わず、必ず再計算）
    comp_cols = _pick_component_cols(df_sess)
    if not comp_cols:
        raise ValueError("構成銘柄の数値列が見つかりません。")

    # 4) 5分グリッドに再整列・ffill → 揃っていない行を除去（末尾の跳ね防止）
    df_aligned = _align_and_filter(df_sess, args.session_start, args.session_end, comp_cols, min_ratio=1.0)
    if df_aligned.empty:
        raise ValueError("全ての行がアライン判定で除外されました。CSVの欠損状況をご確認ください。")

    # 5) 等ウェイト平均から％シリーズを再計算
    series_pct = _compute_index_percent(df_aligned, comp_cols, args.csv_unit)
    if series_pct.empty:
        raise ValueError("指数系列が生成できませんでした。")

    # 6) 終値（％）
    pct_last_percent = float(np.round(series_pct.iloc[-1], 4))
    now_ts_jst = pd.Timestamp.now(tz=JST)

    # 7) 出力
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
