# -*- coding: utf-8 -*-
"""
R-BANK9 intraday index snapshot
等ウェイト / 前日終値比（%）で1日チャートを描画（黒背景・SNS向け）
- 共通JSTグリッドへ整列（09:00–15:30, 5分足）
- 行ごとの被覆率（有効銘柄比率）< TH をドロップしてスパイクを抑制
"""

import os
from typing import List, Dict
from datetime import datetime, date, time, timedelta, timezone

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ===== 設定 =====
JST = timezone(timedelta(hours=9))
OUT_DIR = "docs/outputs"
TICKER_FILE = "docs/tickers_rbank9.txt"

IMG_PATH = os.path.join(OUT_DIR, "rbank9_intraday.png")
CSV_PATH = os.path.join(OUT_DIR, "rbank9_intraday.csv")
POST_PATH = os.path.join(OUT_DIR, "rbank9_post_intraday.txt")
STATS_PATH = os.path.join(OUT_DIR, "rbank9_stats.json")
LAST_RUN_PATH = os.path.join(OUT_DIR, "last_run.txt")

# 取得設定
PERIOD_D = "5d"     # 余裕を持って取得（当日抽出）
INTERVAL = "5m"
SESSION_START = time(9, 0)
SESSION_END = time(15, 30)
MIN_COVERAGE = 0.80  # 行ごとの有効銘柄比率がこれ未満なら採用しない

FIGSIZE = (16, 9)
DPI = 160


# ===== util =====
def jst_now() -> datetime:
    return datetime.now(JST)


def load_tickers(path: str) -> List[str]:
    xs: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            xs.append(s)
    if not xs:
        raise RuntimeError("No tickers found.")
    return xs


def ensure_close_1d(close_like: pd.DataFrame | pd.Series, index) -> pd.Series:
    """
    yfinanceの Close が Series / DataFrame(複数列や重複列) のどちらでも
    1D Series[float] に正規化して返す。
    """
    if isinstance(close_like, pd.Series):
        ser = pd.to_numeric(close_like, errors="coerce").astype(float)
        ser.index = index
        return ser.dropna()

    df = close_like.apply(pd.to_numeric, errors="coerce")
    df = df.loc[:, df.notna().any(axis=0)]
    if df.shape[1] == 0:
        return pd.Series(dtype=float, index=index)

    if df.shape[1] == 1:
        ser = df.iloc[:, 0]
    else:
        # 有効データ点が最大の列を採用
        best = df.count(axis=0).idxmax()
        ser = df[best]
    ser = ser.astype(float)
    ser.index = index
    return ser.dropna()


def fetch_prev_close(ticker: str) -> float:
    d = yf.download(ticker, period="10d", interval="1d",
                    auto_adjust=False, progress=False, prepost=False)
    if d.empty or "Close" not in d.columns:
        raise RuntimeError(f"prev close empty for {ticker}")
    s = ensure_close_1d(d["Close"], d.index)
    if len(s) >= 2:
        return float(s.iloc[-2])
    return float(s.iloc[-1])


def build_jst_grid(the_day: date) -> pd.DatetimeIndex:
    start = datetime.combine(the_day, SESSION_START, tzinfo=JST)
    end = datetime.combine(the_day, SESSION_END, tzinfo=JST)
    # yfinance 5m 足に合わせて 5 分刻み
    return pd.date_range(start=start, end=end, freq="5min", tz=JST)


def fetch_intraday_all(tickers: List[str]) -> Dict[str, pd.Series]:
    """
    全銘柄いっぺんに取得し、JSTへ変換して返す（Series: Close）
    """
    raw = yf.download(
        tickers=" ".join(tickers),
        period=PERIOD_D,
        interval=INTERVAL,
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
        prepost=False,
    )
    out: Dict[str, pd.Series] = {}

    # 単一銘柄だとMultiIndexでなくなるため吸収
    def extract_close(df_or_ser):
        if isinstance(df_or_ser, pd.DataFrame):
            if "Close" in df_or_ser.columns:
                return ensure_close_1d(df_or_ser["Close"], df_or_ser.index)
            return pd.Series(dtype=float)
        # Series の場合はそのまま Close とみなす
        return ensure_close_1d(df_or_ser, df_or_ser.index)

    if isinstance(raw.columns, pd.MultiIndex):
        # 例: ('8035.T','Close') の形
        for t in tickers:
            if (t, "Close") in raw.columns:
                ser = ensure_close_1d(raw[(t, "Close")], raw.index)
            elif t in raw.columns.get_level_values(0):
                # 念のためフォールバック
                ser = extract_close(raw[t])
            else:
                ser = pd.Series(dtype=float)
            if not ser.empty:
                # UTC -> JST
                idx = pd.to_datetime(ser.index)
                if idx.tz is None:
                    idx = idx.tz_localize("UTC")
                ser.index = idx.tz_convert(JST)
                out[t] = ser
    else:
        # 単一銘柄ケース
        ser = extract_close(raw)
        if not ser.empty:
            idx = pd.to_datetime(ser.index)
            if idx.tz is None:
                idx = idx.tz_localize("UTC")
            ser.index = idx.tz_convert(JST)
            out[tickers[0]] = ser

    return out


# ===== index build =====
def build_index_prev_close(tickers: List[str]) -> pd.DataFrame:
    # 当日（JST）
    today_jst = jst_now().date()
    grid = build_jst_grid(today_jst)

    # 前日終値を個別取得
    prev_map: Dict[str, float] = {}
    for t in tickers:
        prev_map[t] = fetch_prev_close(t)

    # 日中全量
    series_map = fetch_intraday_all(tickers)

    # 当日だけにトリムし、共通グリッドへ reindex
    df_list = []
    for t in tickers:
        ser = series_map.get(t)
        if ser is None or ser.empty:
            continue
        ser = ser[(ser.index.date == today_jst)]
        if ser.empty:
            continue
        # グリッドへ整列
        ser = ser.reindex(grid).astype(float)
        df_list.append(ser.rename(t))

    if not df_list:
        raise RuntimeError("当日の有効データが取得できませんでした。")

    px = pd.concat(df_list, axis=1)  # price matrix (time x tickers)

    # 前日終値比 (%)
    for t in px.columns:
        pc = prev_map.get(t)
        if pc and pc != 0:
            px[t] = (px[t] / pc - 1.0) * 100.0
        else:
            px[t] = pd.NA

    # 被覆率で行をフィルタ（十分な銘柄が揃っていない時刻を除外）
    valid_count = px.notna().sum(axis=1)
    need = max(1, int(len(px.columns) * MIN_COVERAGE + 1e-9))
    mask = valid_count >= need
    px = px[mask]
    if px.empty:
        raise RuntimeError("被覆率フィルタ後にデータが残りませんでした。")

    # 等加重平均
    px["R_BANK9"] = px.mean(axis=1, skipna=True)

    # CSV 保存用に time と列を整える
    px_out = px.copy()
    px_out.index.name = "datetime_jst"
    px_out.to_csv(CSV_PATH, encoding="utf-8")

    return px


# ===== plot / outputs =====
def pick_line_color(v: float) -> str:
    return "#00e5d7" if v >= 0 else "#ff4d4d"


def plot_series(series: pd.Series) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    last_val = float(series.iloc[-1])
    c = pick_line_color(last_val)

    plt.close("all")
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    # スタイル
    for sp in ax.spines.values():
        sp.set_color("#444444")
    ax.tick_params(colors="white")

    # ゼロライン
    ax.axhline(0, color="#666666", linewidth=1.0)

    # 陰陽フィル
    y = series.values
    x = series.index
    ax.fill_between(x, 0, y, where=(y >= 0), alpha=0.25, color=c)
    ax.fill_between(x, 0, y, where=(y < 0), alpha=0.25, color="#7b2e43")

    # ライン
    ax.plot(x, y, color=c, linewidth=2.6, label="R-BANK9")

    ax.set_title(
        f"R-BANK9 Intraday Snapshot ({jst_now().strftime('%Y/%m/%d %H:%M')})",
        color="white", fontsize=22, pad=12
    )
    ax.set_xlabel("Time", color="white")
    ax.set_ylabel("Change vs Prev Close (%)", color="white")
    ax.legend(facecolor="black", edgecolor="#444444", labelcolor="white", loc="upper left")

    fig.tight_layout()
    plt.savefig(IMG_PATH, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_post_text(pct_last: float) -> None:
    sign = "🔺" if pct_last >= 0 else "🔻"
    with open(POST_PATH, "w", encoding="utf-8") as f:
        f.write(
            f"{sign} R-BANK9 日中スナップショット（{jst_now().strftime('%Y/%m/%d %H:%M')}）\n"
            f"{pct_last:+.2f}%（前日終値比）\n"
            f"※ 構成9銘柄の等ウェイト\n"
            f"#地方銀行 #R_BANK9 #日本株\n"
        )


def save_stats(pct_last: float) -> None:
    # サイト側が比率(=小数)なら×100して表示できるよう unit を ratio で渡す
    obj = {
        "index_key": "R_BANK9",
        "label": "R-BANK9",
        "pct_intraday": pct_last / 100.0,   # ratio
        "unit": "ratio",
        "basis": "prev_close",
        "session": {
            "start": "09:00",
            "end": "15:30",
            "anchor": "09:00",
        },
        "updated_at": jst_now().isoformat(),
    }
    import json
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    tickers = load_tickers(TICKER_FILE)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[INFO] Build intraday index (prev_close basis) ...")
    px = build_index_prev_close(tickers)
    series = px["R_BANK9"]

    # 出力
    last = float(series.iloc[-1])
    plot_series(series)
    save_post_text(last)
    save_stats(last)

    with open(LAST_RUN_PATH, "w", encoding="utf-8") as f:
        f.write(jst_now().strftime("%Y/%m/%d %H:%M:%S %Z"))

    print(f"[INFO] Done. Last = {last:+.2f}%")

if __name__ == "__main__":
    main()
