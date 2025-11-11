#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
R-BANK9 intraday snapshot/post generator (robust)
入力: docs/outputs/rbank9_intraday.csv
出力: PNG / TXT / JSON（指数リポ内に一時出力。ワークフローでサイトへコピー）
"""

from pathlib import Path
import argparse
import json
import re
from datetime import datetime, time, timedelta
import io  # ← 追加（pandas.compat ではなく io.StringIO を使う）
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import pytz

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-text", required=True)
    ap.add_argument("--snapshot-png", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--index-key", required=True)
    ap.add_argument("--session-start", default="09:00")
    ap.add_argument("--session-end",   default="15:30")
    ap.add_argument("--day-anchor",    default="09:00")
    ap.add_argument("--basis",         default="prev_close")
    ap.add_argument("--value-type",    default="auto")
    return ap.parse_args()

TZ_JST = pytz.timezone("Asia/Tokyo")

def _read_intraday_csv(csv_path: Path) -> pd.DataFrame:
    """CSV を頑健に読む（コメント/空白/BOM を除去、ts/pct を正規化）"""
    raw = Path(csv_path).read_text(encoding="utf-8-sig")
    lines = []
    for ln in raw.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        ln = ln.replace("，", ",")
        lines.append(ln)
    if not lines:
        return pd.DataFrame(columns=["ts", "pct"])

    head = lines[0].lower()
    if not (head.startswith("ts") and "pct" in head):
        lines = ["ts,pct"] + lines

    tmp = "\n".join(lines)

    # ここを io.StringIO に修正
    df = pd.read_csv(
        io.StringIO(tmp),
        engine="python",
        comment="#"
    )

    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    if "ts" not in df.columns or "pct" not in df.columns:
        return pd.DataFrame(columns=["ts", "pct"])

    def _to_num(x):
        if pd.isna(x):
            return None
        s = str(x).strip()
        s = s.replace("％", "").replace("%", "")
        s = re.sub(r"[^\d\.\-\+eE]", "", s)
        try:
            return float(s)
        except Exception:
            return None

    df["pct"] = df["pct"].map(_to_num)
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True).dt.tz_convert(TZ_JST)
    df = df.dropna(subset=["ts", "pct"]).sort_values("ts").reset_index(drop=True)
    return df[["ts", "pct"]]

def _filter_today_session(df: pd.DataFrame, session_start: str, session_end: str, day_anchor: str) -> pd.DataFrame:
    now = datetime.now(TZ_JST)
    a_h, a_m = map(int, day_anchor.split(":"))
    anchor = now.replace(hour=a_h, minute=a_m, second=0, microsecond=0)
    base_date = (anchor - timedelta(days=1)).date() if now < anchor else anchor.date()
    s_h, s_m = map(int, session_start.split(":"))
    e_h, e_m = map(int, session_end.split(":"))
    s_dt = TZ_JST.localize(datetime.combine(base_date, time(s_h, s_m)))
    e_dt = TZ_JST.localize(datetime.combine(base_date, time(e_h, e_m)))
    return df.loc[(df["ts"] >= s_dt) & (df["ts"] <= e_dt)].copy()

def _ensure_dark(ax):
    fig = ax.figure
    fig.patch.set_facecolor("#0B1220")
    ax.set_facecolor("#0B1220")
    for spine in ax.spines.values():
        spine.set_color("#C8D0E0")
    ax.tick_params(colors="#C8D0E0")
    ax.yaxis.label.set_color("#C8D0E0")
    ax.xaxis.label.set_color("#C8D0E0")
    ax.title.set_color("#E8ECF3")
    ax.grid(True, color="#2A3448", alpha=0.6, linewidth=0.8)

def _plot(df: pd.DataFrame, out_png: Path, title: str):
    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    _ensure_dark(ax)
    if df.empty:
        ax.set_title(f"{title} (no data)")
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color="#93A1B5", fontsize=16)
    else:
        ax.plot(df["ts"], df["pct"], linewidth=2.0, color="#5EC8E5")
        ax.set_title(title)
        ax.xaxis.set_major_formatter(DateFormatter("%H:%M", tz=TZ_JST))
        ax.set_ylabel("Change vs Prev Close (%)")
        ax.set_xlabel("Time")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def main():
    args = parse_args()
    in_csv = Path(args.csv)
    out_png = Path(args.snapshot_png)
    out_txt = Path(args.out_text)
    out_json = Path(args.out_json)

    df_all = _read_intraday_csv(in_csv)
    df = _filter_today_session(df_all, args.session_start, args.session_end, args.day_anchor)
    latest_pct = float(df["pct"].iloc[-1]) if not df.empty else 0.0

    _plot(df, out_png, f"{args.label} Intraday Snapshot (JST)")

    now_jst = datetime.now(TZ_JST).strftime("%Y/%m/%d %H:%M")
    sign = "+" if latest_pct >= 0 else ""
    if df.empty:
        post = f"▲ {args.label} 日中スナップショット（no data）\n+0.00%（基準: {args.basis}）\n#R_BANK9 #日本株\n"
    else:
        post = f"▲ {args.label} 日中スナップショット（{now_jst} JST）\n{sign}{latest_pct:.2f}%（基準: {args.basis}）\n#R_BANK9 #日本株\n"
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text(post, encoding="utf-8")

    payload = {
        "index_key": args.index_key,
        "label": args.label,
        "pct_intraday": latest_pct,
        "basis": args.basis,
        "session": {
            "start": args.session_start,
            "end": args.session_end,
            "anchor": args.day_anchor
        },
        "updated_at": datetime.now(TZ_JST).isoformat()
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
