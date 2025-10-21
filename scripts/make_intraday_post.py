#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- helpers ----------

def to_jst_index(raw: pd.DataFrame, dt_col: str) -> pd.DataFrame:
    if dt_col not in raw.columns:
        raise ValueError(f"CSVに対象列 '{dt_col}' がありません。列={list(raw.columns)}")
    dt = pd.to_datetime(raw[dt_col], errors="coerce", utc=True)
    # すでにtz付きなら tz_convert、無ければ tz_localize
    if pd.api.types.is_datetime64tz_dtype(dt):
        idx = dt.tz_convert("Asia/Tokyo")
    else:
        idx = dt.dt.tz_localize("UTC").dt.tz_convert("Asia/Tokyo")
    out = raw.drop(columns=[dt_col])
    out.index = idx
    return out.sort_index()

def clip_session(df: pd.DataFrame, start_hm: str, end_hm: str) -> pd.DataFrame:
    s_h, s_m = map(int, start_hm.split(":"))
    e_h, e_m = map(int, end_hm.split(":"))
    mask = (df.index.time >= pd.Timestamp(hour=s_h, minute=s_m).time()) & \
           (df.index.time <= pd.Timestamp(hour=e_h, minute=e_m).time())
    clipped = df.loc[mask]
    if clipped.empty:
        raise ValueError(f"セッション内データがありません。指定範囲: {start_hm}–{end_hm} JST / "
                         f"データ範囲: {df.index.min()} – {df.index.max()}")
    return clipped

def percentize(series: pd.Series, value_type: str) -> pd.Series:
    """
    入力 series を「％（パーセント）」に正規化して返す。
    value_type:
      - 'auto'    : 自動判別
      - 'percent' : 既に％（例 3.2 = +3.2%）
      - 'ratio'   : 比率（例 1.032 = +3.2%）
      - 'decimal' : 小数（例 0.032 = +3.2%）
    """
    s = pd.to_numeric(series, errors="coerce")
    s = s.dropna()

    if s.empty:
        return s

    vt = (value_type or "auto").lower()

    if vt == "auto":
        sample = s.tail(min(60, len(s))).abs()
        med = sample.median()
        # ざっくり閾値で判定：※すべて％に変換して返す
        if med > 20:        # すでに％表記（例 3.2 = +3.2%）
            return s
        elif med > 0.2:     # ratio 近辺（例 0.98〜1.05）
            # ただし 0.5〜2.0 のような「水準」もここに入る可能性があるが、
            # intraday指数で 1.x のレンジは ratio と見なすのが自然
            return (s - 1.0) * 100.0
        else:               # 0.03 など decimal
            return s * 100.0

    if vt == "percent":
        return s
    if vt == "ratio":
        return (s - 1.0) * 100.0
    if vt in ("decimal", "fraction"):
        return s * 100.0

    # 未知指定は自動扱い
    return percentize(s, "auto")

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-key", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-text", required=True)
    ap.add_argument("--snapshot-png", required=True)
    ap.add_argument("--session-start", default="09:00")
    ap.add_argument("--session-end", default="15:30")
    ap.add_argument("--day-anchor", default="09:00")     # ラベル用
    ap.add_argument("--basis", default="prev_close")     # 出力メタ用
    ap.add_argument("--value-type", default="auto")      # auto / percent / ratio / decimal
    ap.add_argument("--dt-col", default="Datetime")      # 日時列名
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSVが見つかりません: {csv_path}")

    raw = pd.read_csv(csv_path)
    df = to_jst_index(raw, args.dt_col)

    # value列の推定：index_key があれば最優先、なければ最後の数値列
    value_col = args.index_key if args.index_key in df.columns else None
    if value_col is None:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            raise ValueError("数値列が見つかりません。")
        value_col = num_cols[-1]

    # セッション抽出
    df_sess = clip_session(df[[value_col]], args.session_start, args.session_end)
    series_raw = df_sess[value_col]

    # ％に正規化
    pct_series = percentize(series_raw, args.value_type)

    # 最新値（％）
    latest_pct = float(pct_series.iloc[-1])

    # ---------- 描画 ----------
    plt.close("all")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    ax.plot(pct_series.index, pct_series.values, linewidth=2)

    title_label = (args.label or args.index_key).replace("_", "-")
    ts_label = pct_series.index.tz_convert("Asia/Tokyo") if hasattr(pct_series.index, "tz") else pct_series.index
    ts_str = ts_label[-1].strftime("%Y/%m/%d %H:%M")
    ax.set_title(f"{title_label} Intraday Snapshot ({ts_str})", pad=12)
    ax.set_ylabel("Change vs Prev Close (%)")
    ax.set_xlabel("Time")
    ax.grid(True, alpha=0.25)

    # 余白/フレームの最小化
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    for spine in ax.spines.values():
        spine.set_visible(False)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_color("white")
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_color("white")
    ax.title.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.xaxis.label.set_color("white")

    fig.tight_layout()
    fig.savefig(args.snapshot_png, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    # ---------- 出力 ----------
    # テキスト
    sign = "▲" if latest_pct >= 0 else "▼"
    label = args.label or args.index_key
    out_text = f"{sign} {label} 日中スナップショット（{ts_str}）\n{latest_pct:+.2f}%（基準: prev_close）\n#{label.replace(' ', '')} #日本株\n"
    Path(args.out_text).write_text(out_text, encoding="utf-8")

    # JSON（ダッシュボード等で参照）
    payload = {
        "index_key": args.index_key,
        "label": label,
        "pct_intraday": latest_pct,
        "basis": "prev_close",
        "session": {"start": args.session_start, "end": args.session_end, "anchor": args.day_anchor},
        "updated_at": ts_label[-1].isoformat(),
    }
    # 標準のjsonでOK（pandas.io.jsonは使いません）
    import json
    Path(args.out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
