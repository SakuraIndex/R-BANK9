#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
R-BANK9 intraday snapshot/post generator (robust)

優先順位:
1) docs/outputs/rbank9_intraday.csv が ts,pct ならそのまま採用
2) 同CSVが ts,val(レベル) なら、history から前日終値を取り % 化して採用
3) それでも時系列が作れない場合は、最新レベルを推定し % のみ算出
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

TZ = "Asia/Tokyo"

# ---------- dark theme ----------
BG   = "#0b1220"
FG   = "#d1d5db"
GRID = "#334155"

def _set_dark(ax):
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(colors=FG, labelcolor=FG)
    ax.grid(True, alpha=0.35, color=GRID, linestyle="-", linewidth=0.7)
    ax.title.set_color(FG)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    plt.rcParams["figure.facecolor"] = BG

# ---------- CSV 読み ----------
def _clean_lines(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        s = line.strip().replace("\ufeff", "")
        if not s or s.startswith("#"):
            continue
        if s.count(",") != 1:
            continue
        out.append(s)
    return out

def _read_ts_val_like(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame(columns=["ts","val"])
    raw = csv_path.read_text(encoding="utf-8", errors="ignore")
    lines = _clean_lines(raw)
    if not lines:
        return pd.DataFrame(columns=["ts","val"])
    buf = io.StringIO("\n".join(lines))
    df = pd.read_csv(buf, header=None, names=["ts","val"])
    ts  = pd.to_datetime(df["ts"], errors="coerce", utc=True).dt.tz_convert(TZ)
    val = pd.to_numeric(df["val"], errors="coerce")
    df  = pd.DataFrame({"ts": ts, "val": val}).dropna().sort_values("ts").reset_index(drop=True)
    return df

def _detect_is_pct(series: pd.Series) -> bool:
    s = series.dropna()
    if s.empty: return False
    inside = ((s >= -30) & (s <= 30)).mean()
    return bool(inside >= 0.8)

def _read_prev_close(history_csv: Optional[Path]) -> Optional[float]:
    if not history_csv or not history_csv.exists():
        return None
    raw = history_csv.read_text(encoding="utf-8", errors="ignore")
    lines = _clean_lines(raw)
    if not lines:
        return None
    df = pd.read_csv(io.StringIO("\n".join(lines)), header=None, names=["date","value"])
    d  = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    v  = pd.to_numeric(df["value"], errors="coerce")
    df = pd.DataFrame({"date": d, "value": v}).dropna().sort_values("date")
    if df.empty: return None
    today = pd.Timestamp.now(tz=TZ).date()
    prev = df[df["date"].dt.date < today]
    if prev.empty:
        prev = df.iloc[:-1] if len(df) >= 2 else df
    return float(prev.iloc[-1]["value"]) if not prev.empty else None

def _guess_close_level(index_key: str, outputs_dir: Path) -> Optional[float]:
    csv = outputs_dir / f"{index_key.lower()}_intraday.csv"
    if not csv.exists(): return None
    df = _read_ts_val_like(csv)
    if df.empty: return None
    today = pd.Timestamp.now(tz=TZ).date()
    dd = df[df["ts"].dt.date == today]
    if dd.empty: dd = df
    return float(dd.iloc[-1]["val"])

def _to_pct_from_level(df_level: pd.DataFrame, prev_close: Optional[float]) -> pd.DataFrame:
    if prev_close is None or df_level.empty:
        return pd.DataFrame(columns=["ts","pct"])
    pct = (df_level["val"] / float(prev_close) - 1.0) * 100.0
    out = pd.DataFrame({"ts": df_level["ts"], "pct": pct})
    return out.dropna().sort_values("ts").reset_index(drop=True)

# ---------- 可視化 ----------
def _render(df_pct: pd.DataFrame, out_png: Path, label: str):
    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    _set_dark(ax)
    title = f"{label} Intraday Snapshot (JST)"
    if df_pct.empty:
        ax.set_title(title + " (no data)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Change vs Prev Close (%)")
    else:
        ax.plot(df_pct["ts"], df_pct["pct"], linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Change vs Prev Close (%)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def _latest(df_pct: pd.DataFrame) -> Optional[float]:
    return None if df_pct.empty else float(df_pct.iloc[-1]["pct"])

def _build_post(label: str, pct: Optional[float], basis: str) -> str:
    stamp = pd.Timestamp.now(tz=TZ).strftime("%Y/%m/%d %H:%M")
    if pct is None:
        return f"▲ {label} 日中スナップショット（no data）\n+0.00%（基準: {basis}）\n#R_BANK9 #日本株\n"
    sign = "+" if pct >= 0 else ""
    return f"▲ {label} 日中スナップショット（{stamp} JST）\n{sign}{pct:.2f}%（基準: {basis}）\n#R_BANK9 #日本株\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",          required=True)
    ap.add_argument("--history-csv",  default="docs/outputs/rbank9_history.csv")
    ap.add_argument("--index-key",    required=True)
    ap.add_argument("--label",        required=True)
    ap.add_argument("--basis",        default="prev_close")
    ap.add_argument("--session-start", default="09:00")
    ap.add_argument("--session-end",   default="15:30")
    ap.add_argument("--day-anchor",    default="09:00")
    ap.add_argument("--value-type",    default="auto")
    ap.add_argument("--out-json",     required=True)
    ap.add_argument("--out-text",     required=True)
    ap.add_argument("--snapshot-png", required=True)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    # ★ ここを修正： history-csv → history_csv
    hist_csv = Path(args.history_csv) if args.history_csv else None
    out_png  = Path(args.snapshot_png)
    out_txt  = Path(args.out_text)
    out_json = Path(args.out_json)

    d = _read_ts_val_like(csv_path)

    df_pct = pd.DataFrame(columns=["ts","pct"])
    latest_pct_val: Optional[float] = None

    if not d.empty:
        if _detect_is_pct(d["val"]):
            df_pct = d.rename(columns={"val":"pct"})
        else:
            prev_close = _read_prev_close(hist_csv)
            df_pct = _to_pct_from_level(d, prev_close)

    if df_pct.empty:
        prev_close = _read_prev_close(hist_csv)
        guessed = _guess_close_level(args.index_key, Path("docs/outputs"))
        if prev_close is not None and guessed is not None:
            latest_pct_val = (float(guessed) / float(prev_close) - 1.0) * 100.0

    _render(df_pct, out_png, args.label)
    latest_pct_val = _latest(df_pct) if not df_pct.empty else latest_pct_val

    post = _build_post(args.label, latest_pct_val, args.basis)
    out_txt.write_text(post, encoding="utf-8")

    out_json.write_text(
        json.dumps(
            {
                "index_key": args.index_key,
                "label": args.label,
                "pct_intraday": 0.0 if latest_pct_val is None else latest_pct_val,
                "basis": args.basis,
                "session": {"start": args.session_start, "end": args.session_end, "anchor": args.day_anchor},
                "updated_at": pd.Timestamp.now(tz=TZ).isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

if __name__ == "__main__":
    main()
