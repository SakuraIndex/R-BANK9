#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
R-BANK9 intraday snapshot generator (dark theme + inf-safe)
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

JST = "Asia/Tokyo"

# ====== JST index utilities ======
def to_jst_index(df: pd.DataFrame, dt_col: str) -> pd.DataFrame:
    if dt_col not in df.columns:
        raise ValueError(f"Column '{dt_col}' not found. Available={list(df.columns)}")

    dt = pd.to_datetime(df[dt_col], errors="coerce")
    if dt.isna().all():
        raise ValueError(f"Datetime conversion failed for column '{dt_col}'")

    is_tz_aware = getattr(dt.dtype, "tz", None) is not None
    if is_tz_aware:
        dt = dt.dt.tz_convert(JST)
    else:
        dt = dt.dt.tz_localize(JST)

    out = df.copy()
    out.index = dt
    out.drop(columns=[dt_col], inplace=True, errors="ignore")
    out = out[~out.index.duplicated(keep="last")]
    out.sort_index(inplace=True)
    return out


# ====== Series helpers ======
def pick_series(df: pd.DataFrame, index_key: str) -> pd.Series:
    candidates = [
        index_key, index_key.upper(), index_key.lower(),
        index_key.replace("-", "_"), index_key.replace("-", "_").upper(),
        f"{index_key}_mean", f"{index_key.upper()}_mean",
    ]
    for c in candidates:
        if c in df.columns:
            return df[c].astype(float)

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) == 1:
        return df[num_cols[0]].astype(float)

    raise ValueError(f"No valid numeric column found for key='{index_key}'")


def session_filter(s: pd.Series, start_hm: str, end_hm: str) -> pd.Series:
    st_h, st_m = map(int, start_hm.split(":"))
    ed_h, ed_m = map(int, end_hm.split(":"))
    t0 = s.index.tz_convert(JST)
    mask = (
        (t0.hour > st_h) | ((t0.hour == st_h) & (t0.minute >= st_m))
    ) & (
        (t0.hour < ed_h) | ((t0.hour == ed_h) & (t0.minute <= ed_m))
    )
    return s[mask]


def compute_change(
    s: pd.Series, basis: str = "prev_close", value_type: str = "ratio"
) -> Tuple[pd.Series, float]:
    if s.empty:
        raise ValueError("No intraday data in selected session.")

    ref = s.iloc[0]
    if ref == 0 or np.isnan(ref):
        ref = np.nanmean(s.values)
    if ref == 0 or np.isnan(ref):
        ref = 1.0  # fallback

    ratio = (s / ref) - 1.0
    ratio.replace([np.inf, -np.inf], np.nan, inplace=True)
    ratio.fillna(0.0, inplace=True)

    if value_type == "percent":
        series = ratio * 100.0
        last_pct = float(series.iloc[-1])
    else:
        series = ratio
        last_pct = float(ratio.iloc[-1] * 100.0)

    if not np.isfinite(last_pct):
        last_pct = 0.0

    return series, last_pct


# ====== Save functions ======
def save_json(out_path: Path, payload: dict):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload["pct_intraday"] = float(payload.get("pct_intraday", 0.0))
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_post_text(out_path: Path, label: str, last_pct: float, basis: str, now_jst: pd.Timestamp):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sign = "▲" if last_pct >= 0 else "▼"
    txt = f"{sign} {label} 日中スナップショット（{now_jst.strftime('%Y/%m/%d %H:%M')}）\n" \
          f"{last_pct:+.2f}%（基準: {basis}）\n" \
          f"#{label.replace('-', '').upper()} #日本株\n"
    out_path.write_text(txt, encoding="utf-8")


# ====== Plot function ======
def save_plot(out_path: Path, series: pd.Series, title: str, label: str):
    plt.style.use("dark_background")

    fig, ax = plt.subplots(figsize=(11, 6), dpi=120)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    # 線と色
    ax.plot(series.index.tz_convert(JST), series.values, linewidth=2.2, color="cyan", label=label)

    # 枠線削除
    for spine in ax.spines.values():
        spine.set_visible(False)

    # グリッドと軸
    ax.grid(True, color="gray", alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_xlabel("Time", color="white")
    ax.set_ylabel("Change vs Prev Close (%)", color="white")
    ax.tick_params(colors="white")
    ax.legend(loc="upper left", facecolor="black", edgecolor="none", labelcolor="white")
    ax.set_title(title, color="white", pad=10)

    plt.tight_layout(pad=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, facecolor="black", bbox_inches="tight", pad_inches=0.05)
    plt.close()


# ====== Main ======
@dataclass
class Args:
    index_key: str
    csv: str
    out_json: str
    out_text: str
    snapshot_png: str
    session_start: str
    session_end: str
    day_anchor: str
    basis: str
    value_type: str
    dt_col: str
    label: str


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
    p.add_argument("--value-type", default="ratio", choices=["ratio", "percent"])
    p.add_argument("--dt-col", default="Datetime")
    p.add_argument("--label", default=None)
    a = p.parse_args()
    return Args(
        index_key=a.index_key,
        csv=a.csv,
        out_json=a.out_json,
        out_text=a.out_text,
        snapshot_png=a.snapshot_png,
        session_start=a.session_start,
        session_end=a.session_end,
        day_anchor=a.day_anchor,
        basis=a.basis,
        value_type=a.value_type,
        dt_col=a.dt_col,
        label=a.label or a.index_key.replace("_", "-")
    )


def main():
    args = parse_args()

    raw = pd.read_csv(args.csv)
    df = to_jst_index(raw, args.dt_col)
    series_raw = pick_series(df, args.index_key)

    series = session_filter(series_raw, args.session_start, args.session_end)
    if series.empty:
        raise ValueError(f"No data between {args.session_start}–{args.session_end} JST.")

    change_series, last_pct = compute_change(series, basis=args.basis, value_type=args.value_type)

    now_jst = pd.Timestamp.now(tz=JST)
    payload = {
        "index_key": args.index_key,
        "label": args.label,
        "pct_intraday": last_pct,
        "basis": args.basis,
        "session": {"start": args.session_start, "end": args.session_end, "anchor": args.day_anchor},
        "updated_at": now_jst.isoformat(),
    }

    save_json(Path(args.out_json), payload)
    save_post_text(Path(args.out_text), args.label, last_pct, args.basis, now_jst)

    # Y軸は常に%で表示
    y = change_series * (100.0 if args.value_type == "ratio" else 1.0)
    title = f"{args.label} Intraday Snapshot ({now_jst.strftime('%Y/%m/%d %H:%M')})"
    save_plot(Path(args.snapshot_png), y, title=title, label=args.label)


if __name__ == "__main__":
    main()
