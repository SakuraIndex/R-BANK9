#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
R-BANK9 intraday snapshot/post generator (site charts)
- 入力CSV: docs/outputs/rbank9_intraday.csv
    フォーマットは次のいずれかを許容:
      A) ヘッダ2列: ts,pct
         例) 2025-11-11T09:05:00+09:00,0.12
      B) 複数列ヘッダ (先頭1行が銘柄見出しの装飾、2行目に 'ts,pct' )
         例) ",5830.T, ... # R_BANK9" などの行を含む → 自動で 'ts,pct' だけ抽出

- 出力（サイト側にコピーされる前段の中間物）:
    docs/outputs/rbank9_intraday.png
    docs/outputs/rbank9_post_intraday.txt
    docs/outputs/rbank9_stats.json
"""

from __future__ import annotations

from pathlib import Path
import io
import json
import re
from typing import Tuple

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, AutoDateLocator

# ---------- 入出力 ----------
IN_CSV   = Path("docs/outputs/rbank9_intraday.csv")
OUT_PNG  = Path("docs/outputs/rbank9_intraday.png")
OUT_TXT  = Path("docs/outputs/rbank9_post_intraday.txt")
OUT_STAT = Path("docs/outputs/rbank9_stats.json")

INDEX_KEY = "R_BANK9"
LABEL     = "R-BANK9"
BASIS     = "prev_close"

# ---------- スタイル（サイトのダークに合わせる） ----------
BG = "#0b1320"         # 背景
FG = "#c7d0e0"         # 文字/軸
LINE = "#3fd2ff"       # ライン
GRID = "#2a3550"       # グリッド

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "savefig.edgecolor": BG,
    "axes.edgecolor": BG,        # 外枠（白い線）を見えなくする
    "axes.labelcolor": FG,
    "xtick.color": FG,
    "ytick.color": FG,
    "text.color": FG,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.alpha": 0.6,
    "grid.linestyle": "-",
    "grid.linewidth": 0.7,
})

TITLE = "R-BANK9 Intraday Snapshot (JST)"


# ---------- CSV 読み取り（堅牢化） ----------

def _load_raw_csv_text(p: Path) -> str:
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")
    txt = p.read_text(encoding="utf-8", errors="ignore")
    # BOM/全角空白/末尾空行 などを整理
    txt = txt.replace("\ufeff", "").strip()
    return txt


def _extract_ts_pct_from_mixed(csv_text: str) -> pd.DataFrame:
    """
    混在ファイルに対応:
      - 先頭行が銘柄一覧/装飾等でも、後続に 'ts,pct' 行があればそれを採用
      - コメント的な行（先頭 '#', '//'）は除去
    """
    # コメント/空行の除去
    lines = []
    for raw in csv_text.splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith("#") or s.startswith("//"):
            continue
        lines.append(s)

    # まずは素直に2列CSVとして読んでみる
    try:
        df_try = pd.read_csv(io.StringIO("\n".join(lines)))
        # 列名正規化
        cols = [c.strip().lower() for c in df_try.columns]
        if "ts" in cols and "pct" in cols:
            df = df_try.rename(columns={df_try.columns[cols.index("ts")]: "ts",
                                        df_try.columns[cols.index("pct")]: "pct"})
            return df[["ts", "pct"]]
    except Exception:
        pass

    # 先頭行が多列だったケース:
    # 2行目が "ts,pct" 以降の単純2列だったら、その部分だけ再読込
    for i, s in enumerate(lines[:10]):  # 冒頭〜10行程度を走査
        if re.match(r"^ts\s*,\s*pct\s*$", s, flags=re.I):
            # 2列ヘッダ行以降を読み直す
            segment = "\n".join(lines[i:])
            df2 = pd.read_csv(io.StringIO(segment))
            df2.columns = [c.strip().lower() for c in df2.columns]
            df2 = df2.rename(columns={"ts": "ts", "pct": "pct"})
            return df2[["ts", "pct"]]

    # どうしても判別できなければ失敗
    raise ValueError("CSV 形式が判別できません（ts,pct が見つからない）。")


def _read_intraday_csv(p: Path) -> pd.DataFrame:
    raw = _load_raw_csv_text(p)
    df = _extract_ts_pct_from_mixed(raw)

    # 型変換
    # タイムスタンプ: 文字列→datetime（JST）
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True).dt.tz_convert("Asia/Tokyo")
    # 数値: "0.12" などを float に。NaN は落とす
    df["pct"] = pd.to_numeric(df["pct"], errors="coerce")

    df = df.dropna(subset=["ts", "pct"]).sort_values("ts").reset_index(drop=True)
    return df


# ---------- 描画 ----------

def _plot_no_data(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=150)
    ax.set_title(TITLE, pad=12)
    ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, color=FG)
    ax.set_xlabel("Time")
    ax.set_ylabel("Change vs Prev Close (%)")
    # 外枠/余白を目立たせない
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0.6)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def _plot_series(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=150)

    ax.plot(df["ts"], df["pct"], linewidth=2.0, color=LINE)

    ax.set_title(TITLE, pad=12)
    ax.set_xlabel("Time")
    ax.set_ylabel("Change vs Prev Close (%)")

    # 軸スケールや日付フォーマット
    ax.xaxis.set_major_locator(AutoDateLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))

    # 外枠の白線を消す
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout(pad=0.6)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


# ---------- メイン ----------

def main() -> None:
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

    try:
        df = _read_intraday_csv(IN_CSV)
    except Exception as e:
        # 取り込み失敗 → no data 画像 & 0%（理由は post に記載）
        _plot_no_data(OUT_PNG)
        OUT_TXT.write_text(
            f"▲ {LABEL} 日中スナップショット (no data)\n"
            f"+0.00%（基準: {BASIS}）\n"
            f"#{INDEX_KEY} #日本株\n",
            encoding="utf-8"
        )
        OUT_STAT.write_text(json.dumps({
            "index_key": INDEX_KEY,
            "label": LABEL,
            "pct_intraday": 0.0,
            "basis": BASIS,
            "session": {"start": "09:00", "end": "15:30", "anchor": "09:00"},
            "updated_at": pd.Timestamp.now(tz="Asia/Tokyo").isoformat(),
            "note": f"csv_error: {type(e).__name__}: {e}"
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    if df.empty:
        _plot_no_data(OUT_PNG)
        last_pct = 0.0
    else:
        _plot_series(df, OUT_PNG)
        last_pct = float(df["pct"].iloc[-1])

    sign = "+" if last_pct >= 0 else ""
    OUT_TXT.write_text(
        f"▲ {LABEL} 日中スナップショット ({pd.Timestamp.now(tz='Asia/Tokyo').strftime('%Y/%m/%d %H:%M JST')})\n"
        f"{sign}{last_pct:.2f}%（基準: {BASIS}）\n"
        f"#{INDEX_KEY} #日本株\n",
        encoding="utf-8"
    )

    OUT_STAT.write_text(json.dumps({
        "index_key": INDEX_KEY,
        "label": LABEL,
        "pct_intraday": last_pct,
        "basis": BASIS,
        "session": {"start": "09:00", "end": "15:30", "anchor": "09:00"},
        "updated_at": pd.Timestamp.now(tz="Asia/Tokyo").isoformat()
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
