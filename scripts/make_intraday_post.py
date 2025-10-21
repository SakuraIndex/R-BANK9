#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
R-BANK9 intraday: make snapshot chart + post text + stats json

修正ポイント
- CSVは「各銘柄の前日終値比[%]」が入っている前提。
  行方向に等加重平均して指数の騰落率[%] series を作る。
- タイムゾーンはCSVの日時が UTC なら JST へ変換。
  tz-naive の場合は UTC とみなしてローカライズ → JST へ。
- セッション抽出は 「データ中の最新日時の“日付(JST)”」を基準に
  start/end を同日内の 09:00–15:30 などで切り出し。
- 画像はダーク背景。凡例（白枠）は出さない。
- 最終時点の騰落率[%] を小数2桁で TXT/JSON に出力。
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
    session_start: str  # "HH:MM"
    session_end: str    # "HH:MM"
    day_anchor: str     # "HH:MM"
    basis: str          # "prev_close" etc.
    value_type: str     # "percent" | "ratio"（※本スクリプトでは percent を出力）
    dt_col: str         # e.g. "Unnamed: 0"
    label: str          # "R-BANK9"


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
    p.add_argument("--value-type", default="percent")  # ここは常に percent を期待
    p.add_argument("--dt-col", default="Unnamed: 0")
    p.add_argument("--label", default="R-BANK9")
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
    )


# ---------- utils ----------

def _to_jst_index(raw: pd.DataFrame, dt_col: str) -> pd.DataFrame:
    """CSVの日時列を index にし、JST の DatetimeIndex へ変換して返す。"""
    # 1) 読み込み
    df = pd.read_csv(
        raw,  # type: ignore[arg-type]
        parse_dates=[dt_col],
        index_col=dt_col,
    )

    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("Datetime index が取得できません。")

    # 2) tz 取り扱い（UTC とみなす）
    if idx.tz is None:
        # naive → UTC ローカライズ → JST
        idx = idx.tz_localize("UTC").tz_convert(JST)
    else:
        # 既に tz 付きなら JST へ変換
        idx = idx.tz_convert(JST)

    df.index = idx

    # 3) 数値化（%が入っている列たち。coerce で非数は NaN）
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def _session_slice(df_jst: pd.DataFrame, start_hm: str, end_hm: str) -> pd.DataFrame:
    """データの『最新時刻の日付(JST)』を基準にセッション時間帯を切り出す。"""
    if df_jst.empty:
        return df_jst

    last_day = df_jst.index[-1].date()  # JST の date
    start = pd.Timestamp(f"{last_day} {start_hm}", tz=JST)
    end = pd.Timestamp(f"{last_day} {end_hm}", tz=JST)

    return df_jst.loc[(df_jst.index >= start) & (df_jst.index <= end)]


def _equal_weight_percent(df_session: pd.DataFrame) -> pd.Series:
    """
    行方向に等加重平均。
    ※CSVは「各銘柄の 前日比[%]」が入っている前提なので、
      そのまま平均（%）を取れば指数の 前日比[%]。
    """
    if df_session.empty:
        return pd.Series(dtype=float)

    # 全行・全列で平均（NaN は除外）
    series_pct = df_session.mean(axis=1, skipna=True)

    # あり得ないスパイク（±50% など）を無視したい場合は下を有効化
    # series_pct = series_pct.clip(lower=-20.0, upper=20.0)

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
    payload = {
        "index_key": index_key,
        "label": label,
        "pct_intraday": float(pct_last),  # 例: 0.36 (= +0.36%)
        "basis": basis,
        "session": {
            "start": start_hm,
            "end": end_hm,
            "anchor": anchor_hm,
        },
        "updated_at": now_ts_jst.isoformat(),
    }
    # 標準 json を使用（pandas.io.json は使わない）
    import json
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _plot_series(path: Path, label: str, series_pct: pd.Series) -> None:
    """
    ダーク背景・凡例なし・枠線スッキリで保存。
    y: Change vs Prev Close (%)
    """
    if series_pct.empty:
        # 空でも生成（空プロット）しておく
        plt.figure(figsize=(12, 6), dpi=150)
        plt.close()
        Path(path).touch()
        return

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

    # 背景
    fig.patch.set_facecolor("#000000")
    ax.set_facecolor("#000000")

    # ライン
    ax.plot(series_pct.index, series_pct.values, linewidth=2.0)

    # 軸・スパイン
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("#444444")
        ax.spines[side].set_linewidth(0.8)

    ax.set_ylabel("Change vs Prev Close (%)")
    ax.set_xlabel("Time")

    # 凡例は表示しない（白枠防止）
    # ax.legend(loc="best", frameon=False)  # ←使用しない

    # タイトル
    last_ts = pd.to_datetime(series_pct.index[-1]).tz_convert(JST)
    ax.set_title(f"{label} Intraday Snapshot ({last_ts.strftime('%Y/%m/%d %H:%M')})")

    # マージン・保存
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)


# ---------- main ----------

def main() -> int:
    args = parse_args()

    print("=== Generate R–BANK9 intraday snapshot ===")
    print(f"CSV file: {args.csv}")
    print(f"Datetime column: {args.dt_col}")
    print(f"VALUE_TYPE(default=percent) requested: {args.value_type}")

    # 1) 読み & JST index 化
    df_jst = _to_jst_index(args.csv, args.dt_col)
    print(f"[make_intraday_post] using columns: {list(df_jst.columns)}")
    print(f"[make_intraday_post] rows(JST)={len(df_jst)}")

    # 2) セッション切り出し
    df_sess = _session_slice(df_jst, args.session_start, args.session_end)
    if df_sess.empty:
        raise ValueError(
            f"セッション内データがありません。"
            f" 指定範囲: {args.session_start}–{args.session_end} JST / "
            f"データ範囲: {df_jst.index.min()} – {df_jst.index.max()}"
        )

    # 3) 等加重 平均[%]
    series_pct = _equal_weight_percent(df_sess)

    # 4) 出力値（最終値の %）
    pct_last = float(np.round(series_pct.iloc[-1], 4))  # 例: +0.36 → 0.36

    now_ts_jst = pd.Timestamp.now(tz=JST)

    # 5) 画像
    _plot_series(Path(args.snapshot_png), args.label, series_pct)
    print(f"[make_intraday_post] snapshot saved -> {args.snapshot_png}")

    # 6) テキスト
    _save_text(Path(args.out_text), args.label, pct_last, args.basis, now_ts_jst)
    print(f"[make_intraday_post] text saved -> {args.out_text}")

    # 7) JSON
    _save_json(Path(args.out_json), args.index_key, args.label, pct_last,
               args.basis, args.session_start, args.session_end, args.day_anchor, now_ts_jst)
    print(f"[make_intraday_post] json saved -> {args.out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
