#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
桜Index - 長期チャート自動生成スクリプト（出来高・移動平均線対応）
────────────────────────────
入力:
  docs/outputs/{key}_intraday.csv  （or *_intraday.txt）
出力:
  docs/outputs/{key}_{7d,1m,1y}.png
  docs/outputs/{key}_{7d,1m,1y}.csv
"""

import os
import re
from datetime import datetime, timedelta, timezone
import pandas as pd
import matplotlib.pyplot as plt

JST = timezone(timedelta(hours=9))
plt.rcParams['font.family'] = 'Noto Sans CJK JP'

# === 設定 ===
SMA_WINDOWS = [5, 25, 75]  # 移動平均線の日数
VOLUME_COLUMN_CANDIDATES = ["volume", "vol", "出来高"]


def log(msg): print(f"[long_charts] {msg}")


def find_input(base, key):
    for name in [f"{key}_intraday.csv", f"{key}_intraday.txt", f"{key}.csv"]:
        path = os.path.join(base, name)
        if os.path.exists(path):
            return path
    return None


def read_data(path):
    """CSV/TXTから時系列データを抽出（time, value, volume）"""
    df = pd.read_csv(path)
    # 列名正規化
    df.columns = [c.lower().strip() for c in df.columns]
    t_candidates = [c for c in df.columns if c in ("time", "timestamp", "date", "datetime")]
    v_candidates = [c for c in df.columns if c in ("close", "price", "value", "index")]
    vol_candidates = [c for c in df.columns if c in VOLUME_COLUMN_CANDIDATES]
    if not t_candidates or not v_candidates:
        # 最低限2列は (time, value)
        df.columns = ["time", "value"] + list(df.columns[2:])
    tcol, vcol = t_candidates[0] if t_candidates else "time", v_candidates[0] if v_candidates else "value"
    volcol = vol_candidates[0] if vol_candidates else None

    # 時刻parse
    def parse_time(x):
        if pd.isna(x): return pd.NaT
        s = str(x)
        if re.fullmatch(r"\d{10}", s):
            return datetime.fromtimestamp(int(s), tz=JST)
        try:
            t = pd.to_datetime(s)
            if t.tzinfo is None:
                t = t.tz_localize(JST)
            return t.tz_convert(JST)
        except Exception:
            return pd.NaT

    df["time"] = df[tcol].apply(parse_time)
    df["value"] = pd.to_numeric(df[vcol], errors="coerce")
    if volcol:
        df["volume"] = pd.to_numeric(df[volcol], errors="coerce")
    else:
        df["volume"] = 0
    df = df.dropna(subset=["time", "value"]).sort_values("time")
    return df[["time", "value", "volume"]]


def to_daily(df):
    """日次データ化（終値＋出来高合計）"""
    df["date"] = df["time"].dt.date
    daily = (
        df.groupby("date", as_index=False)
          .agg({"value": "last", "volume": "sum"})
    )
    daily["time"] = pd.to_datetime(daily["date"]).dt.tz_localize(JST)
    return daily[["time", "value", "volume"]].sort_values("time")


def plot_chart(df, key, label):
    """価格 + 移動平均線 + 出来高"""
    if df.empty:
        log(f"skip empty {key}_{label}")
        return

    # 移動平均線追加
    for w in SMA_WINDOWS:
        df[f"SMA{w}"] = df["value"].rolling(window=w).mean()

    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax2 = ax1.twinx()

    # 出来高バー
    ax2.bar(df["time"], df["volume"], width=0.8, color="gray", alpha=0.3, label="Volume")
    ax2.set_ylabel("Volume", color="gray")
    ax2.tick_params(axis="y", colors="gray")
    ax2.set_ylim(bottom=0)

    # 価格線
    ax1.plot(df["time"], df["value"], color="#ff99cc", lw=1.6, label="Index")
    colors = ["#80d0ff", "#ffd580", "#b0ffb0"]
    for i, w in enumerate(SMA_WINDOWS):
        ax1.plot(df["time"], df[f"SMA{w}"], lw=1.0, color=colors[i], label=f"SMA{w}")

    ax1.set_title(f"{key.upper()} ({label})", color="#ffb6c1")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Index Value")

    fig.tight_layout()
    out_png = f"docs/outputs/{key}_{label}.png"
    plt.legend(loc="upper left")
    plt.savefig(out_png, dpi=160)
    plt.close()
    log(f"saved chart: {out_png}")


def main():
    key = os.environ.get("INDEX_KEY")
    if not key:
        raise SystemExit("ERROR: INDEX_KEY not set")

    base = "docs/outputs"
    src = find_input(base, key)
    if not src:
        raise SystemExit(f"ERROR: input not found under {base}")

    raw = read_data(src)
    daily = to_daily(raw)

    now = datetime.now(tz=JST)
    ranges = {
        "7d": now - timedelta(days=7),
        "1m": now - timedelta(days=31),
        "1y": now - timedelta(days=365),
    }

    for label, since in ranges.items():
        sub = daily[daily["time"] >= since].copy()
        sub.to_csv(f"docs/outputs/{key}_{label}.csv", index=False)
        plot_chart(sub, key, label)


if __name__ == "__main__":
    main()
