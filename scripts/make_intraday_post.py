#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Intraday snapshot & post generator

- 入力 CSV から指定列の intraday 系列を取り出し、前日終値比 or 指定アンカー比の推移を描画/集計
- CSV がレベル（指数値）か、すでに % / 比率の「前日比」なのかを自動判定 or 明示指定
- 生成物:
  - PNG グラフ: --snapshot-png
  - テキスト:   --out-text
  - JSON 統計:  --out-json （pct_intraday は % 数値）
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import math

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

JST = "Asia/Tokyo"


# ---------- CLI ----------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--index-key", required=True, help="Index key label (e.g., R_BANK9)")
    p.add_argument("--csv", required=True, help="Input CSV path")
    p.add_argument("--out-json", required=True, help="Output JSON path")
    p.add_argument("--out-text", required=True, help="Output text path")
    p.add_argument("--snapshot-png", required=True, help="Output PNG path")
    p.add_argument("--session-start", required=True, help="HH:MM JST")
    p.add_argument("--session-end", required=True, help="HH:MM JST")
    p.add_argument("--day-anchor", required=True, help="HH:MM JST label anchor")
    p.add_argument("--basis", default="prev_close", choices=["prev_close", "open@09:00"],
                   help="Change basis: prev_close | open@09:00")
    p.add_argument("--value-type", default="percent", choices=["percent", "ratio"],
                   help="How to *report* values: percent(%) or ratio(=1.0 is +100%). Internals unify to percent.")
    p.add_argument("--dt-col", default="Datetime",
                   help="Datetime column name in CSV. If missing, tries first column as datetime index.")
    p.add_argument("--label", default="", help="Series legend label (for chart legend)")
    # 追加: 入力の意味を制御
    p.add_argument("--assume-change", default="auto",
                   choices=["auto", "level", "pct", "ratio"],
                   help=("Meaning of the CSV data column for index-key: "
                         "level=レベル値, pct=すでに前日比(%) , ratio=すでに前日比(比率), auto=自動判定"))
    return p


# ---------- TZ helpers ----------
def to_jst_index(df: pd.DataFrame, dt_col: str) -> pd.DataFrame:
    if dt_col in df.columns:
        ts = pd.to_datetime(df[dt_col], errors="coerce", utc=True)
        # もし tz-naive が混在するなら一括で UTC とみなす
        if ts.dt.tz is None:
            ts = pd.to_datetime(df[dt_col], errors="coerce").dt.tz_localize("UTC")
        df = df.drop(columns=[dt_col])
        df.index = ts.dt.tz_convert(JST)
    else:
        # インデックスが時刻（文字列/naive/aware）で来るケース
        idx = pd.to_datetime(df.index, errors="coerce")
        if hasattr(idx, "tz") and idx.tz is None:
            idx = idx.tz_localize("UTC")
        idx = idx.tz_convert(JST)
        df.index = idx
    # sort + dropna index
    df = df[~df.index.isna()].sort_index()
    return df


# ---------- 判定/変換 ----------
def looks_like_pct(series: pd.Series) -> bool:
    """すでに %（例: -10〜+10 程度のレンジが多い）っぽいか"""
    s = series.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return False
    mx = float(s.abs().quantile(0.98))
    # 前日比%として自然と思えるレンジ（±100%以内がほとんど）
    return mx <= 100.0

def looks_like_ratio(series: pd.Series) -> bool:
    """すでに比率（-0.3〜0.3 など 1.0 で +100% にするタイプ）っぽいか"""
    s = series.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return False
    mx = float(s.abs().quantile(0.98))
    # 比率は通常 ±1.0 以内（±100%）に収まることが多い（例: 0.05=+5%）
    return mx <= 1.0

def as_percent(series: pd.Series, assume: str) -> pd.Series:
    """
    入力 series を「%値（-5.0 = -5%）」に正規化して返す。
    assume:
      - level : レベル値（指数/価格） -> そのままでは%にならない（後で基準と合わせて%化する）
      - pct   : すでに% -> そのまま
      - ratio : すでに比率 -> ×100
      - auto  : 列名/レンジから推定
    """
    name = series.name or ""
    if assume == "auto":
        # 列名ヒント
        lname = name.lower()
        if any(k in lname for k in ["pct", "%", "percent"]):
            assume_eff = "pct"
        elif any(k in lname for k in ["ratio", "ret", "return"]):
            assume_eff = "ratio"
        else:
            # 数値レンジから推定
            if looks_like_ratio(series):
                assume_eff = "ratio"
            elif looks_like_pct(series):
                assume_eff = "pct"
            else:
                assume_eff = "level"
    else:
        assume_eff = assume

    if assume_eff == "pct":
        return series.astype(float)
    if assume_eff == "ratio":
        return series.astype(float) * 100.0
    # level は % にまだしていない（後段で基準と組み合わせ）
    series.attrs["__as_level__"] = True
    return series.astype(float)


# ---------- セッション抽出 ----------
@dataclass
class Session:
    start: pd.Timestamp
    end: pd.Timestamp

def parse_hm_jst(hm: str, date: pd.Timestamp) -> pd.Timestamp:
    h, m = map(int, hm.split(":"))
    return pd.Timestamp(year=date.year, month=date.month, day=date.day, hour=h, minute=m, tz=JST)

def filter_session(df_jst: pd.DataFrame, start_hm: str, end_hm: str) -> tuple[pd.DataFrame, Session]:
    if df_jst.empty:
        raise ValueError("入力データが空です。")
    day = df_jst.index[-1].normalize().tz_convert(JST)
    start = parse_hm_jst(start_hm, day)
    end   = parse_hm_jst(end_hm,   day)
    out = df_jst[(df_jst.index >= start) & (df_jst.index <= end)].copy()
    if out.empty:
        raise ValueError(f"セッション内データがありません。指定範囲: {start_hm}–{end_hm} JST / "
                         f"データ範囲: {df_jst.index.min()} – {df_jst.index.max()}")
    return out, Session(start, end)


# ---------- メイン計算 ----------
def compute_change_pct(series: pd.Series, basis: str, anchor_hm: str) -> pd.Series:
    """
    series: すでに as_percent() 済み（%）か、attrs['__as_level__']=True のレベル値
    basis:  prev_close | open@09:00
    戻り値: 常に %（float）
    """
    if series.attrs.get("__as_level__", False):
        # レベル -> % へ変換
        s = series.astype(float)
        if basis == "prev_close":
            # 前場の最初の点（または 09:00 直後）を「前日終値近傍」とみなすのではなく、
            # 当日最初の観測からの相対を採用（前日終値にぴったり一致しない指数でも暴れにくい）
            ref = s.iloc[0]
        else:  # open@09:00
            # 09:00 以降で最初に観測できた値
            # （minute 欠損があっても最初の点を anchor として使う）
            ref = s.iloc[0]
        eps = 1e-12
        change_pct = (s / max(ref, eps) - 1.0) * 100.0
        return change_pct
    else:
        # すでに % 値（または ratio から%化済み）
        # prev_close / open@09:00 いずれでも「入力がすでに前日比」であれば、そのまま扱う
        return series.astype(float)


# ---------- 出力 ----------
def render_plot(idx_label: str,
                series_pct: pd.Series,
                png_path: Path,
                legend_label: str,
                title_stamp: str) -> None:
    fig = plt.figure(figsize=(12, 6), dpi=150)
    ax = plt.gca()
    ax.plot(series_pct.index, series_pct.values, linewidth=2.0)
    ax.set_title(f"{idx_label} Intraday Snapshot ({title_stamp})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Change vs Basis (%)")
    # 余白の白縁を消す（要望）
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend([legend_label or idx_label], loc="upper left")
    ax.grid(False)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path)
    plt.close(fig)


def write_text(out_path: Path, label: str, pct_now: float, basis: str, now_jst: pd.Timestamp):
    sgn = "▲" if pct_now >= 0 else "▼"
    out = [
        f"{sgn} {label} 日中スナップショット（{now_jst.strftime('%Y/%m/%d %H:%M')}）",
        f"{pct_now:+.2f}%（基準: {basis}）",
        f"#{label} #日本株",
    ]
    out_path.write_text("\n".join(out), encoding="utf-8")


def write_json(out_path: Path, index_key: str, label: str, pct_now: float,
               basis: str, sess: Session, now_jst: pd.Timestamp):
    payload = {
        "index_key": index_key,
        "label": label or index_key,
        "pct_intraday": float(pct_now),  # 常に % 数値
        "basis": basis,
        "session": {"start": sess.start.strftime("%H:%M"),
                    "end":   sess.end.strftime("%H:%M"),
                    "anchor": "09:00"},
        "updated_at": now_jst.isoformat(),
    }
    out_path.write_text(pd.io.json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------- main ----------
def main() -> int:
    ap = build_parser()
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV が見つかりません: {csv_path}", file=sys.stderr)
        return 1

    # CSV 読み込み
    raw = pd.read_csv(csv_path)
    # index-key 列を特定（完全一致が無ければ先頭の非日時列のうち、index-key に近いもの）
    # まずは完全一致 or 大文字小文字無視一致を探す
    col = None
    if args.index_key in raw.columns:
        col = args.index_key
    else:
        # 大文字小文字・ハイフン/アンダースコア等を緩く比較
        def norm(s: str) -> str:
            return "".join(ch for ch in s.upper() if ch.isalnum())
        target = norm(args.index_key)
        for c in raw.columns:
            if norm(str(c)) == target:
                col = c
                break
        if col is None:
            # 「_mean」「_avg」など代表列があれば採用
            for c in raw.columns:
                lc = str(c).lower()
                if any(k in lc for k in ["_mean", "_avg", "composite", "index", "rbank9", "r_bank9"]):
                    col = c
                    break
    if col is None:
        # 最後の砦: 数値列のうち NaN が少ない列を代表に
        num_cols = [c for c in raw.columns if pd.api.types.is_numeric_dtype(raw[c])]
        if num_cols:
            col = num_cols[0]
    if col is None:
        print(f"CSV に対象列 '{args.index_key}' がありません。列={list(raw.columns)}", file=sys.stderr)
        return 1

    # Datetime インデックス化
    df = raw.set_index(args.dt_col) if args.dt_col in raw.columns else raw.set_index(raw.columns[0])
    df = to_jst_index(df, args.dt_col)
    if df.empty:
        print("CSV から JST 時系列を構築できませんでした。", file=sys.stderr)
        return 1

    # セッション抽出
    df_sess, sess = filter_session(df, args.session_start, args.session_end)

    # 値の意味を % に正規化
    series_in = df_sess[col].astype(float)
    series_pct_like = as_percent(series_in, args.assume_change)  # % 値 or レベル (attrs フラグ)

    # % に確定
    series_pct = compute_change_pct(series_pct_like, args.basis, args.day_anchor)
    series_pct = series_pct.replace([np.inf, -np.inf], np.nan).dropna()
    if series_pct.empty:
        print("セッション内に有効データがありません（変換後）。", file=sys.stderr)
        return 1

    # 直近 % 値
    pct_now = float(series_pct.iloc[-1])

    # 出力
    now_jst = pd.Timestamp.now(tz=JST)
    label = (args.label or args.index_key).upper()
    render_plot(label, series_pct, Path(args.snapshot_png), label, now_jst.strftime("%Y/%m/%d %H:%M"))
    write_text(Path(args.out_text), label, pct_now, args.basis, now_jst)
    write_json(Path(args.out_json), args.index_key, label, pct_now, args.basis, sess, now_jst)

    return 0


if __name__ == "__main__":
    sys.exit(main())
