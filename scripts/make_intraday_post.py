# scripts/make_intraday_post.py
# -*- coding: utf-8 -*-
"""
汎用：日中スナップショット PNG とテキスト投稿を生成
・CSV: 1列目 Datetime, 2列目が対象指数（列名は大文字小文字を無視）
・--basis:
    - open@HH:MM  : 当日 HH:MM の価格を基準に騰落率
    - prev_close  : 前日終値を基準に騰落率
・--session-start/--session-end/--day-anchor : 表示やラベル用
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

JST = "Asia/Tokyo"

# ---------- helpers ----------
def to_ts_jst(s: pd.Series) -> pd.DatetimeIndex:
    dt = pd.to_datetime(s, utc=True, errors="coerce")
    if dt.dt.tz is None:
        dt = dt.dt.tz_localize("UTC")
    return dt.dt.tz_convert(JST)

def find_value_column(df: pd.DataFrame, index_key: str) -> str:
    cols = {c.lower(): c for c in df.columns}
    low = index_key.lower()
    if low in cols: 
        return cols[low]
    # 2列目をフォールバック（先頭はDatetime想定）
    if df.shape[1] >= 2:
        return df.columns[1]
    raise ValueError(f"CSVに対象列が見つかりません（index_key={index_key}）")

def filter_session(df: pd.DataFrame, start_hhmm: str, end_hhmm: str) -> pd.DataFrame:
    s_h, s_m = map(int, start_hhmm.split(":"))
    e_h, e_m = map(int, end_hhmm.split(":"))
    mask = (
        (df.index.hour > s_h) | ((df.index.hour == s_h) & (df.index.minute >= s_m))
    ) & (
        (df.index.hour < e_h) | ((df.index.hour == e_h) & (df.index.minute <= e_m))
    )
    return df.loc[mask]

def pick_anchor_value(df: pd.DataFrame, basis: str, anchor_hhmm: str, value_col: str) -> float:
    if basis == "prev_close":
        # 当日0時以前の直近を「前日終値相当」として採用（夜間データが無い場合は当日の最初）
        prev = df[df.index.date < df.index[-1].date()]
        if not prev.empty:
            return float(prev[value_col].iloc[-1])
        return float(df[value_col].iloc[0])

    if basis.startswith("open@"):
        hhmm = basis.split("@", 1)[1] if "@" in basis else anchor_hhmm
        a_h, a_m = map(int, hhmm.split(":"))
        # アンカー以降の最初の値
        after = df[(df.index.hour > a_h) | ((df.index.hour == a_h) & (df.index.minute >= a_m))]
        if not after.empty:
            return float(after[value_col].iloc[0])
        return float(df[value_col].iloc[0])

    raise ValueError(f"未知のbasis: {basis}")

def compute_change_pct(series: pd.Series, anchor: float) -> pd.Series:
    return (series / anchor - 1.0) * 100.0

def make_plot(df: pd.DataFrame, value_col: str, title: str, out_png: Path):
    plt.figure(figsize=(12, 6), dpi=120)
    ax = plt.gca()
    # ダーク背景 & 枠線無し
    ax.set_facecolor("#0b0b0b")
    plt.gcf().patch.set_facecolor("#0b0b0b")
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.plot(df.index, df["pct"], linewidth=2.5, color="#1dd3c5", label=title)
    ax.legend(loc="upper left", frameon=False, labelcolor="#cfd8dc")
    ax.set_ylabel("Change vs Prev Close (%)", color="#cfd8dc")
    ax.set_xlabel("Time", color="#cfd8dc")
    ax.tick_params(colors="#cfd8dc")
    ax.xaxis.set_major_formatter(DateFormatter("%m-%d %H:%M"))
    ax.grid(False)
    plt.title(title, color="#cfd8dc")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, facecolor="#0b0b0b")
    plt.close()

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-key", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-text", required=True)
    ap.add_argument("--snapshot-png", required=True)
    ap.add_argument("--session-start", required=True)
    ap.add_argument("--session-end", required=True)
    ap.add_argument("--day-anchor", required=True)
    ap.add_argument("--basis", default="open@09:00")   # 例: open@09:00 / prev_close
    ap.add_argument("--value-type", choices=["percent"], default="percent")
    args = ap.parse_args()

    raw = pd.read_csv(args.csv)
    if raw.empty:
        raise ValueError("CSVが空です。")

    # 時刻と列解決
    raw.index = to_ts_jst(raw.iloc[:, 0])
    value_col = find_value_column(raw, args.index_key)

    # 当日セッション抽出
    df = filter_session(raw[[value_col]].copy(), args.session_start, args.session_end)
    if df.empty:
        raise ValueError("セッション内データがありません。")

    # 基準値
    anchor_val = pick_anchor_value(df, args.basis, args.day_anchor, value_col)
    df["pct"] = compute_change_pct(df[value_col].astype(float), anchor_val)

    # 出力
    label = args.index_key.upper() if args.index_key.isupper() else args.index_key.replace("_", "-").upper()
    title = f"{label} Intraday Snapshot"
    make_plot(df, value_col, title, Path(args.snapshot_png))

    pct_now = float(df["pct"].iloc[-1])
    stats = {
        "index_key": label,
        "label": label,
        "pct_intraday": pct_now,
        "basis": args.basis if args.basis != "" else f"open@{args.day_anchor}",
        "session": {"start": args.session_start, "end": args.session_end, "anchor": args.day_anchor},
        "updated_at": pd.Timestamp.now(tz=JST).isoformat(),
    }
    Path(args.out_json).write_text(pd.Series(stats).to_json(force_ascii=False, indent=2))

    sign = "▲" if pct_now >= 0 else "▼"
    Path(args.out_text).write_text(
        f"{sign} {label} 日中スナップショット "
        f"({pd.Timestamp.now(tz=JST).strftime('%Y/%m/%d %H:%M')})\n"
        f"{pct_now:.2f}%（基準: {stats['basis']}）\n"
        f"#{label} #日本株\n",
        encoding="utf-8"
    )

if __name__ == "__main__":
    main()
