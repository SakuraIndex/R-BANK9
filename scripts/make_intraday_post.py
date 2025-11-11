#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

JST = timezone(timedelta(hours=9))

# ====== 設定（必要に応じて引数化してOK） ======
OUT_DIR = Path("docs/outputs")
CSV_PATH = OUT_DIR / "rbank9_intraday.csv"
PNG_PATH = OUT_DIR / "rbank9_intraday.png"
STATS_PATH = OUT_DIR / "stats.json"
POST_PATH = OUT_DIR / "post_intraday.txt"

INDEX_KEY = "R_BANK9"
LABEL = "R-BANK9"
BASIS = "prev_close"  # 表示用

# ====== CSV 読み取り ======
def _read_csv_jst(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        print(f"[warn] CSV not found: {csv_path}")
        return pd.DataFrame(columns=["ts", "pct"])

    try:
        df = pd.read_csv(
            csv_path,
            comment="#",
            encoding="utf-8",
            engine="python",
            skip_blank_lines=True,
            skipinitialspace=True,
        )
    except Exception as e:
        print(f"[warn] read_csv failed ({e}). Returning empty dataframe.")
        return pd.DataFrame(columns=["ts", "pct"])

    # 列名ゆらぎの許容
    cols = {c.strip().lower(): c for c in df.columns}
    ts_col = cols.get("ts") or cols.get("time") or cols.get("timestamp")
    pct_col = cols.get("pct") or cols.get("percent") or cols.get("change") or cols.get("ret")

    if ts_col is None or pct_col is None:
        print("[warn] required columns not found. Returning empty dataframe.")
        return pd.DataFrame(columns=["ts", "pct"])

    # 正規化
    df = df[[ts_col, pct_col]].rename(columns={ts_col: "ts", pct_col: "pct"})
    # 時刻はUTC推定→JSTに変換（CSVが+09:00を含む場合も to_datetime が吸収）
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True).dt.tz_convert(JST)
    df["pct"] = pd.to_numeric(df["pct"], errors="coerce")

    df = df.dropna(subset=["ts", "pct"]).sort_values("ts").reset_index(drop=True)
    return df

# ====== 可視化 ======
def _plot(df: pd.DataFrame, out_png: Path, title: str):
    fig, ax = plt.subplots(figsize=(9.5, 3.6), dpi=160)

    # ダークテーマ風
    bg = "#0b1420"
    grid = "#1f2a37"
    txt = "#cbd5e1"
    ax.set_facecolor(bg)
    fig.patch.set_facecolor(bg)
    for spine in ax.spines.values():
        spine.set_color(grid)
    ax.tick_params(colors=txt)
    ax.grid(True, color=grid, alpha=0.45, linewidth=0.6)

    ax.set_xlabel("Time", color=txt)
    ax.set_ylabel("Change vs Prev Close (%)", color=txt)

    if df.empty:
        ax.text(0.5, 0.5, "no data", color=txt, ha="center", va="center",
                transform=ax.transAxes, alpha=0.8, fontsize=12)
    else:
        # x軸はnaiveにして表示
        x = df["ts"].dt.tz_convert(JST).dt.tz_localize(None)
        y = df["pct"]
        ax.plot(x, y, linewidth=1.8)

    ax.set_title(title, color=txt, fontsize=10, pad=8)
    fig.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

# ====== メイン ======
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = _read_csv_jst(CSV_PATH)

    now = datetime.now(JST)
    title = f"{LABEL} Intraday Snapshot (JST)"

    _plot(df, PNG_PATH, title)

    pct = float(df["pct"].iloc[-1]) if not df.empty else 0.0

    stats = {
        "index_key": INDEX_KEY,
        "label": LABEL,
        "pct_intraday": pct,
        "basis": BASIS,
        "session": {"start": "09:00", "end": "15:30", "anchor": "09:00"},
        "updated_at": now.isoformat(),
    }
    STATS_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    post = (
        f"▲ {LABEL} 日中スナップショット ({now:%Y/%m/%d %H:%M})\n"
        f"{pct:+.2f}%（基準: {BASIS}）\n"
        f"#{INDEX_KEY} #日本株"
    )
    POST_PATH.write_text(post, encoding="utf-8")

    print("[ok] generated:", PNG_PATH, STATS_PATH, POST_PATH)


if __name__ == "__main__":
    main()
