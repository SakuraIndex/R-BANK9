#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Intraday snapshot & post generator

- 読み取り CSV（例: docs/outputs/rbank9_intraday.csv）から、日中セッション(09:00-15:30 JST)の
  指定インデックス列を抜き出し、前日終値基準の騰落率(%)を描画・テキスト化・統計JSON化します。
- 値の種類(value_type):
    - "ratio"   : 前日終値=1.0 を基準とする比率 (例 1.0123 ⇒ +1.23%)
    - "percent" : 前日終値からの騰落率そのもの(%) (例 +1.23 ⇒ +1.23%)
  自動ガード: value_type="ratio" 指定でも、|値| が明らかに % っぽい(>10)場合は percent とみなします。
- 日付列(dt_col)は任意。指定が無い・見つからない場合は最左列が日時列とみなします。
- 余白・白い枠線を完全に排除した黒ベースの図に戻しています。
"""

from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import pytz


JST = pytz.timezone("Asia/Tokyo")


# ---------- helpers ----------

def _debug(msg: str) -> None:
    print(f"[make_intraday_post] {msg}", flush=True)


def to_jst_index(df: pd.DataFrame, dt_col: Optional[str]) -> pd.DataFrame:
    """flexible datetime indexer -> JST DatetimeIndex"""
    cols = list(df.columns)
    # pick dt column
    col = None
    if dt_col and dt_col in cols:
        col = dt_col
    else:
        # よくある候補
        for c in cols:
            lc = str(c).strip().lower()
            if lc in ("datetime", "date", "time", "timestamp") or "date" in lc or "time" in lc:
                col = c
                break
        if col is None:
            # 最左列を日時と仮定
            col = cols[0]

    try:
        dt = pd.to_datetime(df[col], errors="coerce")
    except Exception:
        # mixed 型対策
        dt = pd.to_datetime(df[col].astype(str), errors="coerce")

    if dt.isna().all():
        raise ValueError(f"CSVC '{col}' 列を日時に変換できません。")

    # tz 処理
    if getattr(dt.dt, "tz", None) is None:
        dt = dt.dt.tz_localize(JST)  # naive → JST 付与
    else:
        dt = dt.dt.tz_convert(JST)   # tz aware → JST へ

    out = df.copy()
    out.index = dt
    # drop dt column if it's not the first time-series col
    if col in out.columns:
        out = out.drop(columns=[col])
    out = out.sort_index()
    _debug(f"Datetime column = '{col}', rows = {len(out)} (JST)")
    return out


def find_series(df: pd.DataFrame, index_key: str, label: Optional[str]) -> Tuple[pd.Series, str]:
    """列名ゆるマッチ（R-BANK9 / R_BANK9 / rbank9 など）"""
    def norm(s: str) -> str:
        return "".join(ch for ch in s.upper() if ch.isalnum())

    target = norm(index_key if index_key else (label or ""))
    if not target:
        # 最後の列
        col = df.columns[-1]
        return df[col].astype(float), str(col)

    # 完全一致 → 正規化一致 → 含む
    for col in df.columns:
        if str(col) == index_key:
            return df[col].astype(float), str(col)
    for col in df.columns:
        if norm(str(col)) == target:
            return df[col].astype(float), str(col)
    for col in df.columns:
        if target in norm(str(col)):
            return df[col].astype(float), str(col)

    # fallback: 最後の列
    col = df.columns[-1]
    _debug(f"対象列 '{index_key}' が見つからないため fallback -> '{col}'")
    return df[col].astype(float), str(col)


def filter_session(df: pd.DataFrame, start_hm: str, end_hm: str) -> pd.DataFrame:
    """JSTの同一日タイムレンジでフィルタ"""
    st = pd.to_datetime(start_hm, format="%H:%M").time()
    ed = pd.to_datetime(end_hm, format="%H:%M").time()

    day = df.index[0].date()
    start = pd.Timestamp(day, tz=JST).replace(hour=st.hour, minute=st.minute)
    end = pd.Timestamp(day, tz=JST).replace(hour=ed.hour, minute=ed.minute)

    sub = df[(df.index >= start) & (df.index <= end)]
    if sub.empty:
        raise ValueError(f"セッション内データがありません。指定範囲: {start_hm}–{end_hm} JST / データ範囲: {df.index.min()} – {df.index.max()}")
    return sub


def normalize_to_pct(series: pd.Series, value_type: str) -> Tuple[pd.Series, str]:
    """
    値を '前日終値比(%)' に正規化
      - ratio   : (x - 1.0) * 100
      - percent : そのまま (%)
    ただし ratio 指定でも、|値| が明らかに %（>10）っぽければ自動で percent として扱う。
    """
    vt = (value_type or "ratio").lower()
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan).dropna()

    if vt == "ratio":
        # auto-guard: 値のスケールが%っぽい場合は percent と解釈
        if s.abs().quantile(0.95) > 10:  # 95%点が10%超
            _debug("value_type='ratio' ですが値が % スケールと判断 → percent として扱います")
            vt = "percent"

    if vt == "ratio":
        pct = (s - 1.0) * 100.0
    elif vt == "percent":
        pct = s.astype(float)
    else:
        raise ValueError(f"value_type は 'ratio' か 'percent' を指定してください（指定値: {value_type}）")

    # 物理的にありえない巨大値は NaN に（データ異常対策）
    pct = pct.where(pct.abs() <= 50.0, np.nan)  # ±50% 超は欠損扱い
    pct = pct.dropna()
    if pct.empty:
        raise ValueError("騰落率系列が空になりました。入力の値の種類(value_type)が合っているか確認してください。")
    return pct, vt


def arrow_and_fmt(v: float) -> Tuple[str, str]:
    sign = "▲" if v >= 0 else "▼"
    txt = f"{v:+.2f}%"
    return sign, txt


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dark_ax(ax: plt.Axes) -> None:
    ax.set_facecolor("#000000")
    ax.figure.set_facecolor("#000000")
    for spine in ax.spines.values():
        spine.set_visible(False)  # 白い枠線を完全に消す
    ax.tick_params(colors="#B0B0B0")
    ax.xaxis.label.set_color("#B0B0B0")
    ax.yaxis.label.set_color("#B0B0B0")
    ax.title.set_color("#E0E0E0")
    ax.grid(True, color="#202020", linewidth=0.8, alpha=0.9)


# ---------- main ----------

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
    dt_col: Optional[str]
    label: Optional[str]


def parse_args() -> Args:
    p = argparse.ArgumentParser()
    p.add_argument("--index-key", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-text", required=True)
    p.add_argument("--snapshot-png", required=True)
    p.add_argument("--session-start", required=True)
    p.add_argument("--session-end", required=True)
    p.add_argument("--day-anchor", required=True)
    p.add_argument("--basis", default="prev_close")
    p.add_argument("--value-type", choices=["ratio", "percent"], required=True)
    p.add_argument("--dt-col", default=None)
    p.add_argument("--label", default=None)
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


def main() -> int:
    args = parse_args()

    # --- load CSV
    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")
    raw = pd.read_csv(args.csv)
    _debug(f"CSV file: {args.csv}, rows={len(raw)}, cols={list(raw.columns)}")

    df = to_jst_index(raw, args.dt_col)
    series, used_col = find_series(df, args.index_key, args.label)
    _debug(f"using column = '{used_col}'")

    # --- session slice
    sess_df = filter_session(series.to_frame("val"), args.session_start, args.session_end)
    series_sess = sess_df["val"]

    # --- normalize to percent
    pct_series, decided_type = normalize_to_pct(series_sess, args.value_type)
    _debug(f"decided value_type = {decided_type}")
    # 最後まで欠損が残る可能性に備えて forward-fill（板寄せ等欠損の穴埋め）
    pct_series = pct_series.sort_index().ffill()

    # --- compute display point: latest pct
    latest_ts = pct_series.dropna().index.max()
    latest_pct = float(pct_series.loc[latest_ts])

    # --- figure (dark, no border)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    dark_ax(ax)

    # 線色はデフォルト（環境依存）でOK。凡例を見やすく。
    label = (args.label or args.index_key).upper().replace("_", "-")
    ax.plot(pct_series.index, pct_series.values, linewidth=2.0, label=label)

    ax.set_title(f"{label} Intraday Snapshot ({latest_ts.strftime('%Y/%m/%d %H:%M')})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Change vs Prev Close (%)")
    ax.legend(facecolor="#000000", edgecolor="#000000", labelcolor="#B0B0B0")

    ax.xaxis.set_major_formatter(DateFormatter("%m-%d %H", tz=JST))

    # 余白を極小化（白縁防止）
    plt.margins(x=0)
    plt.tight_layout(pad=0.5)
    args.snapshot_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.snapshot_png, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    _debug(f"snapshot saved -> {args.snapshot_png}")

    # --- post text
    updown, pct_txt = arrow_and_fmt(latest_pct)
    title_line = f"{updown} {label} 日中スナップショット ({latest_ts.strftime('%Y/%m/%d %H:%M')})"
    body_line = f"{pct_txt}（基準: {args.basis}）"
    tags_line = f"#{label} #日本株"
    save_text(args.out_text, [title_line, body_line, tags_line])
    _debug(f"text saved -> {args.out_text}")

    # --- stats json
    payload = {
        "index_key": label.replace("-", "_"),
        "label": label,
        "pct_intraday": float(round(latest_pct, 8)),
        "basis": args.basis,
        "session": {"start": args.session_start, "end": args.session_end, "anchor": args.day_anchor},
        "updated_at": pd.Timestamp.now(tz=JST).isoformat(),
    }
    save_json(args.out_json, payload)
    _debug(f"json saved -> {args.out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
