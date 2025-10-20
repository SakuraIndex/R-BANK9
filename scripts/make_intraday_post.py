#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


JST = "Asia/Tokyo"


def to_jst_index(raw: pd.DataFrame, dt_col: str) -> pd.DataFrame:
    """
    CSVから読んだDataFrameに対し、時刻列(dt_col)をJSTのDatetimeIndexにする。
    - dt_col が tz-naive: tz_localize(JST)
    - dt_col が tz-aware: tz_convert(JST)
    """
    if dt_col not in raw.columns:
        raise ValueError(f"CSVに対象列 '{dt_col}' がありません。列={list(raw.columns)}")

    s = pd.to_datetime(raw[dt_col], errors="coerce", utc=False)
    if s.isna().all():
        raise ValueError(f"CSV列 '{dt_col}' が時刻に変換できません。")

    # tz情報を判定してJSTへ
    tz = getattr(s.dtype, "tz", None)
    if tz is None:
        # naive → JSTとしてローカライズ
        s = s.dt.tz_localize(JST, nonexistent="shift_forward", ambiguous="NaT")
    else:
        # tz-aware → JSTへ変換
        s = s.dt.tz_convert(JST)

    out = raw.copy()
    out.index = s
    out = out.drop(columns=[dt_col])
    # NaT は除外
    out = out[~out.index.isna()].sort_index()
    return out


def filter_session(df_jst: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """
    セッション時間でフィルタ。
    DatetimeIndex は tz-aware(JST)を前提。
    """
    # between_time は tz-aware でもOK（時刻だけで判定）
    try:
        return df_jst.between_time(start_time=start, end_time=end)
    except TypeError:
        # pandas の古い仕様差分対策
        start_h, start_m = [int(x) for x in start.split(":")]
        end_h, end_m = [int(x) for x in end.split(":")]
        mask = ((df_jst.index.hour > start_h) | ((df_jst.index.hour == start_h) & (df_jst.index.minute >= start_m))) & \
               ((df_jst.index.hour < end_h) | ((df_jst.index.hour == end_h) & (df_jst.index.minute <= end_m)))
        return df_jst[mask]


def pick_value_column(df: pd.DataFrame, index_key: str, dt_col: str | None = None) -> str:
    """
    値列の自動選択を堅牢化。
    優先順:
      1) 完全一致 (index_key)
      2) 部分一致 (index_key を含む)
      3) *_mean / *_avg
      4) 最初の非時刻列
    """
    cols = list(df.columns)

    cand = [c for c in cols if c == index_key]
    if not cand:
        cand = [c for c in cols if index_key in c]
    if not cand:
        cand = [c for c in cols if c.lower().endswith("_mean") or c.lower().endswith("_avg")]
    if not cand:
        cand = [c for c in cols if c != (dt_col or "Datetime")]
    if not cand:
        raise ValueError("値列を特定できません。")

    return cand[0]


def compute_change(series: pd.Series, value_type: str) -> pd.Series:
    """
    前日終値基準（またはアンカー基準）での変化率/比を計算。
    現仕様: セッション先頭値を基準値として利用。
    value_type: 'percent' → % 表示, 'ratio' → 比(= 0.035 など)
    """
    base = series.iloc[0]
    if pd.isna(base) or base == 0:
        raise ValueError("基準値が不正です（NaN もしくは 0）。")

    if value_type == "percent":
        return (series / base - 1.0) * 100.0
    else:  # 'ratio'
        return (series / base - 1.0)


def style_ax(ax: plt.Axes) -> None:
    """黒背景・外枠無し・白系目盛・シアン線用グリッドの軽い調整"""
    ax.set_facecolor("#000000")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors="#DDDDDD")
    ax.grid(True, alpha=0.15)


def plot_intraday(ts: pd.Series, title: str, ylabel: str, out_png: Path, label: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 6), dpi=110)
    fig.patch.set_facecolor("#000000")
    style_ax(ax)

    ax.plot(ts.index, ts.values, linewidth=2.0, color="#00E5FF", label=label)
    ax.legend(facecolor="#000000", edgecolor="#000000", labelcolor="#DDDDDD")
    ax.set_title(title, color="#FFFFFF")
    ax.set_xlabel("Time", color="#DDDDDD")
    ax.set_ylabel(ylabel, color="#DDDDDD")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index-key", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-text", required=True)
    p.add_argument("--snapshot-png", required=True)
    p.add_argument("--session-start", required=True)
    p.add_argument("--session-end", required=True)
    p.add_argument("--day-anchor", required=True)
    p.add_argument("--basis", required=True, help="e.g., prev_close or open@09:00")
    p.add_argument("--value-type", required=True, choices=["ratio", "percent"])
    p.add_argument("--dt-col", required=True)
    p.add_argument("--label", required=True)
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSVが見つかりません: {csv_path}")

    raw = pd.read_csv(csv_path)
    df_jst = to_jst_index(raw, args.dt_col)
    df_sess = filter_session(df_jst, args.session_start, args.session_end)

    if df_sess.empty:
        # セッション内にデータが無いときのエラーメッセージを明確化
        smin = df_jst.index.min()
        smax = df_jst.index.max()
        raise ValueError(
            f"セッション内データがありません。指定範囲: {args.session_start}–{args.session_end} JST / "
            f"データ範囲: {smin} – {smax}"
        )

    # 値列の決定（R-BANK9 / ASTRA4 いずれでも耐性あり）
    val_col = pick_value_column(df_sess, args.index_key, dt_col=args.dt_col)
    series = df_sess[val_col].dropna()
    if series.empty:
        raise ValueError(f"値系列 '{val_col}' に有効なデータがありません。")

    # 変化率/比を算出
    chg = compute_change(series, args.value_type)

    # 集計値（最終値を採用）
    latest = float(chg.iloc[-1])
    stats = {
        "index_key": args.index_key,
        "label": args.label,
        "pct_intraday": latest if args.value_type == "percent" else latest * 100.0,  # JSONは%で持つ
        "basis": args.basis,
        "session": {
            "start": args.session_start,
            "end": args.session_end,
            "anchor": args.day_anchor,
        },
        "updated_at": pd.Timestamp.now(tz=JST).isoformat(),
    }

    # 出力
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # 投稿テキスト
    sign = "▲" if latest >= 0 else "▼"
    if args.value_type == "percent":
        line2 = f"{latest:+.2f}%（基準: {args.basis}）"
        ylabel = "Change vs Prev Close (%)" if args.basis == "prev_close" else "Change (%)"
        title = f"{args.label} Intraday Snapshot ({pd.Timestamp.now(tz=JST):%Y/%m/%d %H:%M})"
    else:
        # ratio（例: +0.035 → 3.5%）
        pct = latest * 100.0
        line2 = f"{pct:+.2f}%（基準: {args.basis}）"
        ylabel = "Change vs Prev Close (%)" if args.basis == "prev_close" else "Change (%)"
        title = f"{args.label} Intraday Snapshot ({pd.Timestamp.now(tz=JST):%Y/%m/%d %H:%M})"

    out_lines = [
        f"{sign} {args.label} 日中スナップショット（{pd.Timestamp.now(tz=JST):%Y/%m/%d %H:%M}）",
        f"{line2}",
        f"#{args.index_key.replace('_', '')} #日本株",
    ]
    with open(args.out_text, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))

    # チャート
    plot_intraday(
        chg,
        title=title,
        ylabel=ylabel,
        out_png=Path(args.snapshot_png),
        label=args.label,
    )


if __name__ == "__main__":
    main()
