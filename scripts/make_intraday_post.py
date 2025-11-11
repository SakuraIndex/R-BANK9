#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
R-BANK9 intraday snapshot/post generator
入力: docs/outputs/rbank9_intraday.csv  (ヘッダ: ts,pct / 例: 2025-11-11T09:05:00+09:00,0.12)
出力:
  - docs/outputs/rbank9_intraday.png
  - docs/outputs/rbank9_post_intraday.txt
  - docs/outputs/rbank9_stats.json  （updated_at, pct_intraday を含む）
"""

from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

IN_CSV   = Path("docs/outputs/rbank9_intraday.csv")
OUT_PNG  = Path("docs/outputs/rbank9_intraday.png")
OUT_TXT  = Path("docs/outputs/rbank9_post_intraday.txt")
OUT_STAT = Path("docs/outputs/rbank9_stats.json")

TITLE = "R-BANK9 Intraday Snapshot (JST)"

def _read_csv() -> pd.DataFrame:
    if not IN_CSV.exists():
        raise ValueError("CSV がありません。")

    # 余計な空白・BOM・全角カンマ等も吸収
    txt = IN_CSV.read_text(encoding="utf-8").strip().replace("\ufeff","")
    if not txt:
        raise ValueError("CSV が空です。")

    # 行ごとの左右空白を除去して書き戻し（頑健化）
    clean = "\n".join(line.strip() for line in txt.splitlines() if line.strip())
    IN_CSV.write_text(clean, encoding="utf-8")

    df = pd.read_csv(IN_CSV)

    # 列名ゆらぎを吸収
    cols = {c.lower().strip(): c for c in df.columns}
    if "ts" not in cols or "pct" not in cols:
        raise ValueError("CSV ヘッダは ts,pct を想定しています。")

    # パース
    df = df.rename(columns={cols["ts"]: "ts", cols["pct"]: "pct"})
    df["ts"]  = pd.to_datetime(df["ts"], errors="coerce", utc=True).dt.tz_convert("Asia/Tokyo")
    df["pct"] = pd.to_numeric(df["pct"], errors="coerce")
    df = df.dropna(subset=["ts","pct"]).sort_values("ts")
    return df

def _plot_no_data(fig, ax):
    ax.set_title(TITLE)
    ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Time")
    ax.set_ylabel("Change vs Prev Close (%)")
    ax.grid(True, alpha=0.3)

def _plot_series(df: pd.DataFrame, latest_pct: float):
    fig, ax = plt.subplots(figsize=(12,6), dpi=160)
    if df.empty:
        _plot_no_data(fig, ax)
    else:
        ax.plot(df["ts"], df["pct"])
        ax.set_title(TITLE)
        ax.set_xlabel("Time")
        ax.set_ylabel("Change vs Prev Close (%)")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches="tight")
    plt.close(fig)

def main():
    try:
        df = _read_csv()
    except Exception as e:
        # CSVを読めなければ「no data」画像だけ出す
        OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(12,6), dpi=160)
        _plot_no_data(fig, ax)
        fig.tight_layout()
        fig.savefig(OUT_PNG, bbox_inches="tight")
        plt.close(fig)

        # post_intraday.txt は「0.00%（データなし）」で出す
        OUT_TXT.write_text("▲ R-BANK9 日中スナップショット（no data）\n+0.00%（基準: prev_close）\n#R_BANK9 #日本株\n", encoding="utf-8")
        # stats.json も最低限
        OUT_STAT.write_text(json.dumps({
            "index_key":"R_BANK9",
            "label":"R-BANK9",
            "pct_intraday": 0.0,
            "basis":"prev_close",
            "updated_at": pd.Timestamp.now(tz="Asia/Tokyo").isoformat()
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[warn] {_read_csv.__name__}: {e}")
        return

    latest_pct = float(df["pct"].iloc[-1]) if not df.empty else 0.0

    # 画像
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    _plot_series(df, latest_pct)

    # ポスト文（%表示はそのまま、符号つけ・小数2桁）
    sign = "+" if latest_pct >= 0 else ""
    post = (
        f"▲ R-BANK9 日中スナップショット（{pd.Timestamp.now(tz='Asia/Tokyo').strftime('%Y/%m/%d %H:%M')} JST）\n"
        f"{sign}{latest_pct:.2f}%（基準: prev_close）\n"
        f"#R_BANK9 #日本株\n"
    )
    OUT_TXT.write_text(post, encoding="utf-8")

    # stats.json
    OUT_STAT.write_text(json.dumps({
        "index_key":"R_BANK9",
        "label":"R-BANK9",
        "pct_intraday": latest_pct,
        "basis":"prev_close",
        "session":{
            "start":"09:00","end":"15:30","anchor":"09:00"
        },
        "updated_at": pd.Timestamp.now(tz="Asia/Tokyo").isoformat()
    }, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
