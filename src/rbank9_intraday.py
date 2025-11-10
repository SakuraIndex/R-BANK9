# -*- coding: utf-8 -*-
"""
R-BANK9 intraday index snapshot
等ウェイト / 前日終値比（%）で1日チャートを描画（黒背景・SNS向け）
- 重要修正:
  1) タイムスタンプを JST 当日で正規化（5分足）
  2) 銘柄ごとの前日終値比(%)を同一5分グリッドに整列
  3) 当日内の前方補完（ffill）で軽微な欠損を埋めるが、補完本数に上限を設定
  4) 集計はクオーラム方式（>= floor(n*0.6) が有効な時刻のみ平均）
  5) クオーラム未達の末尾は自動的に落とす（終盤の1銘柄だけの跳ねを排除）
"""

from __future__ import annotations

import os
import json
from typing import List, Dict
from math import ceil
from datetime import datetime, timezone, timedelta

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ===== 基本設定 =====
JST = timezone(timedelta(hours=9))
OUT_DIR = "docs/outputs"
TICKER_FILE = "docs/tickers_rbank9.txt"

IMG_PATH = os.path.join(OUT_DIR, "rbank9_intraday.png")
CSV_PATH = os.path.join(OUT_DIR, "rbank9_intraday.csv")
POST_PATH = os.path.join(OUT_DIR, "rbank9_post_intraday.txt")
STATS_PATH = os.path.join(OUT_DIR, "rbank9_stats.json")

# yfinance 側の取得：日本株は 1m が不安定のことがあるため 5m を採用
INTRA_PERIOD = "7d"
INTRA_INTERVAL = "5m"

# 前方補完の最大本数（5分足で 3 本 = 15 分まで許容）
FFILL_LIMIT = 3

# 集計に必要なクオーラム（過半数より少し強めに 60%）
QUORUM_RATIO = 0.6

# 補助：JP セッションの目安（タイトル等に使用）
SESSION = {"start": "09:00", "end": "15:30", "anchor": "09:00"}


# ===== ユーティリティ =====
def jst_now_str(fmt="%Y/%m/%d %H:%M") -> str:
    return datetime.now(JST).strftime(fmt)


def ensure_outdir() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)


def load_tickers(path: str) -> List[str]:
    xs: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            xs.append(s)
    return xs


def _to_1d_close(df: pd.DataFrame) -> pd.Series:
    """yfinance の Close を 1D Series[float] に正規化"""
    if "Close" not in df.columns:
        raise ValueError("Close column not found")
    close_like = df["Close"]

    if isinstance(close_like, pd.Series):
        ser = pd.to_numeric(close_like, errors="coerce")
        ser = ser.dropna()
        return ser

    # DataFrame：数値化 → 全欠損列 drop → 最多有効列を採用
    _df = close_like.apply(pd.to_numeric, errors="coerce")
    _df = _df.loc[:, _df.notna().any(axis=0)]
    if _df.shape[1] == 0:
        raise ValueError("no numeric close column")
    if _df.shape[1] == 1:
        ser = _df.iloc[:, 0]
    else:
        best_col = _df.count(axis=0).idxmax()
        ser = _df[best_col]
    return ser.astype(float)


def _tz_to_jst_index(s: pd.Series) -> pd.Series:
    idx = pd.to_datetime(s.index)
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert(JST)
    s.index = idx
    return s


def fetch_prev_close(ticker: str) -> float:
    d = yf.download(ticker, period="10d", interval="1d",
                    auto_adjust=False, progress=False)
    if d.empty:
        raise RuntimeError(f"[WARN] prev close empty for {ticker}")
    s = _to_1d_close(d)
    if len(s) >= 2:
        return float(s.iloc[-2])
    return float(s.iloc[-1])


def fetch_intraday_close_today_jst(ticker: str) -> pd.Series:
    d = yf.download(ticker, period=INTRA_PERIOD, interval=INTRA_INTERVAL,
                    auto_adjust=False, progress=False)
    if d.empty:
        raise RuntimeError(f"[WARN] intraday empty for {ticker}")
    s = _to_1d_close(d)
    s = _tz_to_jst_index(s)

    # JST 当日のみ
    last_day = s.index[-1].date()
    s = s[s.index.date == last_day]
    if s.empty:
        raise RuntimeError(f"[WARN] intraday filtered empty for {ticker}")
    return s


def make_5min_grid(index_like: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """インデックスを 5 分境界に合わせて丸め直したグリッドを作る"""
    # yfinance の 5m はすでに 5 分境界だが、保険で round
    t = pd.Series(0, index=index_like)
    t.index = t.index.map(lambda x: x.floor("5min"))
    grid = pd.DatetimeIndex(sorted(t.index.unique()))
    return grid


# ===== 集計ロジック =====
def build_matrix_pct_prevclose(tickers: List[str]) -> pd.DataFrame:
    """
    返り値: 行=5分グリッド（JST 当日）、列=各 ticker の前日比%, 値は ffill 済み（上限あり）
    """
    # まず全銘柄の生データ取得
    raw: Dict[str, pd.Series] = {}
    prev_close: Dict[str, float] = {}
    for t in tickers:
        try:
            print(f"[INFO] Fetch {t}")
            prev_close[t] = fetch_prev_close(t)
            s_close = fetch_intraday_close_today_jst(t)
            raw[t] = s_close
        except Exception as e:
            print(f"[WARN] skip fetch {t}: {e}")

    if not raw:
        raise RuntimeError("No intraday data fetched.")

    # 共通 5 分グリッドを作成（全銘柄の union → 5m 丸め）
    union_index = pd.DatetimeIndex(sorted(pd.Index([])))
    for s in raw.values():
        union_index = union_index.union(s.index)
    grid = make_5min_grid(union_index)

    # 各銘柄をグリッドに合わせて reindex → 前日比% を算出 → 短い穴は ffill（FFILL_LIMIT）
    mat = pd.DataFrame(index=grid, columns=tickers, dtype=float)

    for t in tickers:
        if t not in raw:
            continue
        s = raw[t].reindex(grid)  # 5分グリッドに乗せる（欠損は NaN）
        p0 = prev_close[t]
        pct = (s / p0 - 1.0) * 100.0

        # 当日内の自然な欠損は 3本まで前方補完（昼休みなど長い穴は埋めない）
        pct = pct.ffill(limit=FFILL_LIMIT)
        mat[t] = pct

    return mat


def equal_weight_with_quorum(mat: pd.DataFrame, quorum_ratio: float) -> pd.Series:
    """
    クオーラム(有効セル数 >= ceil(n * ratio))を満たす行だけで等加重平均。
    条件を満たさない行は NaN。末尾の連続 NaN はドロップ。
    """
    n = mat.shape[1]
    quorum = ceil(n * quorum_ratio)

    valid_counts = mat.notna().sum(axis=1)
    ok = valid_counts >= quorum

    series = mat.where(ok).mean(axis=1, skipna=True)

    # 末尾連続 NaN（終盤のクオーラム未達）を落とす
    # （グラフ末尾の不自然な尻上がり・垂直跳ねを抑止）
    # 後ろから走査して最初の非NaNまでを残す
    if series.isna().any():
        # 最後の非NaNの位置
        last_valid_pos = series.last_valid_index()
        if last_valid_pos is not None:
            series = series.loc[:last_valid_pos]

    return series.dropna(how="all")


# ===== 可視化・出力 =====
def pick_line_color(series: pd.Series) -> str:
    return "#00e5d7" if len(series) and float(series.iloc[-1]) >= 0 else "#ff4d4d"


def plot_series(series: pd.Series) -> None:
    ensure_outdir()
    c = pick_line_color(series)

    plt.close("all")
    fig = plt.figure(figsize=(16, 9), dpi=160)
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    for sp in ax.spines.values():
        sp.set_color("#444444")

    ax.plot(series.index, series.values, color=c, linewidth=3.0, label="R-BANK9")
    ax.axhline(0, color="#666666", linewidth=1.0)
    ax.tick_params(colors="white")
    ax.set_title(
        f"R-BANK9 Intraday Snapshot ({jst_now_str('%Y/%m/%d %H:%M')})",
        color="white", fontsize=22, pad=12
    )
    ax.set_xlabel("Time", color="white")
    ax.set_ylabel("Change vs Prev Close (%)", color="white")
    ax.legend(facecolor="black", edgecolor="#444444", labelcolor="white", loc="upper left")
    fig.tight_layout()
    plt.savefig(IMG_PATH, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_csv(series: pd.Series, mat_pct: pd.DataFrame) -> None:
    ensure_outdir()
    # 出力は「指数のみ」を既存互換で。デバッグ用に末尾へ指数列も含める
    df = mat_pct.copy()
    df["R_BANK9"] = series.reindex(df.index)
    df.to_csv(CSV_PATH, index_label="datetime_jst")


def save_post_text(last_pct: float) -> None:
    ensure_outdir()
    sign = "🔺" if last_pct >= 0 else "🔻"
    with open(POST_PATH, "w", encoding="utf-8") as f:
        f.write(
            f"{sign} R-BANK9 日中スナップショット（{jst_now_str()}）\n"
            f"{last_pct:+.2f}%（前日終値比）\n"
            f"※ 構成9銘柄の等ウェイト（クオーラム集計）\n"
            f"#地方銀行 #R_BANK9 #日本株\n"
        )


def save_stats_json(last_pct: float) -> None:
    ensure_outdir()
    stats = {
        "index_key": "R_BANK9",
        "label": "R-BANK9",
        "pct_intraday": float(last_pct) / 100.0,  # サイト側が ratio にも対応しているため 0.076 のように置く
        "basis": "prev_close",
        "session": SESSION,
        "updated_at": datetime.now(JST).isoformat(),
    }
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


# ===== メイン =====
def main():
    tickers = load_tickers(TICKER_FILE)
    if not tickers:
        raise RuntimeError("No tickers found.")

    print("[INFO] Build matrix (prev_close %) ...")
    mat_pct = build_matrix_pct_prevclose(tickers)

    print("[INFO] Equal-weight with quorum ...")
    series = equal_weight_with_quorum(mat_pct, QUORUM_RATIO)
    if series.empty:
        raise RuntimeError("No valid points after quorum filtering.")

    last = float(series.iloc[-1])

    print("[INFO] Save artifacts ...")
    plot_series(series)
    save_csv(series, mat_pct)
    save_post_text(last)
    save_stats_json(last)

    print("[INFO] done.")


if __name__ == "__main__":
    main()
