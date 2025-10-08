import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import pytz
import matplotlib.pyplot as plt

# ==== 汎用関数 ====

def ensure_tz(series, tz: str):
    """シリーズを指定タイムゾーンに変換"""
    if not isinstance(series, pd.Series):
        return series
    return series.dt.tz_localize("UTC").dt.tz_convert(tz) if series.dt.tz is None else series.dt.tz_convert(tz)

def parse_time_any(x, raw_tz: str, display_tz: str):
    """文字列/数値などを時刻に変換"""
    try:
        ts = pd.to_datetime(x, utc=True)
    except Exception:
        ts = pd.NaT
    if ts is pd.NaT:
        return ts
    if ts.tzinfo is None:
        ts = ts.tz_localize(raw_tz)
    return ts.tz_convert(display_tz)

def pick_time_col(cols_lower):
    """時刻列を推定"""
    candidates = ["time", "timestamp", "date", "datetime"]
    for name in candidates:
        if name in cols_lower:
            return name
    fuzzy = [i for i, c in enumerate(cols_lower) if ("time" in c) or ("date" in c)]
    return cols_lower[fuzzy[0]] if fuzzy else None

def pick_value_col(df):
    """value列の推定"""
    for c in df.columns:
        if "value" in c.lower() or "index" in c.lower() or "score" in c.lower():
            return c
    return None

def pick_volume_col(df):
    """volume列の推定"""
    for c in df.columns:
        if "volume" in c.lower():
            return c
    return None


# ==== 主関数 ====

def read_any(path: Optional[str], raw_tz: str, display_tz: str) -> pd.DataFrame:
    """
    列名を正規化してから、(1) time/value[/volume] 形式  または (2) 銘柄ごとの板状形式 を受け付ける。
    返り値は必ず ["time","value","volume"] を持つ DataFrame。
    """
    if not path:
        return pd.DataFrame(columns=["time", "value", "volume"])

    df = pd.read_csv(path)

    # 列名を小文字に正規化
    raw_cols = list(df.columns)
    cols_lower = [str(c).strip().lower() for c in raw_cols]
    df.columns = cols_lower

    # ---- 時刻列の推定
    tcol = pick_time_col(cols_lower)
    if tcol is None:
        raise KeyError(f"No time-like column found. columns={list(df.columns)}")

    # ---- time/value/volume 形式（value/volume が単一列で存在）
    vcol = pick_value_col(df)
    volcol = pick_volume_col(df)
    if vcol in df.columns and (volcol in df.columns or volcol is None or volcol == 0):
        out = pd.DataFrame()
        out["time"] = df[tcol].apply(lambda x: parse_time_any(x, raw_tz, display_tz))
        out["time"] = ensure_tz(out["time"], display_tz)
        out["value"] = pd.to_numeric(df[vcol], errors="coerce")
        out["volume"] = (
            pd.to_numeric(df[volcol], errors="coerce") if (volcol and volcol in df.columns) else 0
        )
        out = out.dropna(subset=["time", "value"]).sort_values("time").reset_index(drop=True)
        return out

    # ---- 盤面形式: 時刻列以外の「数値にできる列」を等加重平均して value を作る
    num_cols: list[str] = []
    for c in df.columns:
        if c == tcol:
            continue
        as_num = pd.to_numeric(df[c], errors="coerce")
        if as_num.notna().sum() > 0:
            num_cols.append(c)

    if len(num_cols) == 0:
        # 数値化できる列がない場合は空
        return pd.DataFrame(columns=["time", "value", "volume"])

    out = pd.DataFrame()
    out["time"] = df[tcol].apply(lambda x: parse_time_any(x, raw_tz, display_tz))
    out["time"] = ensure_tz(out["time"], display_tz)

    # 列ごとに数値化してから平均（バグ修正ポイント）
    vals_df = df[num_cols].apply(lambda s: pd.to_numeric(s, errors="coerce"))
    out["value"] = vals_df.mean(axis=1)

    out["volume"] = 0
    out = out.dropna(subset=["time", "value"]).sort_values("time").reset_index(drop=True)
    return out


# ==== メイン ====

def main():
    # 例: intraday_path として最新のCSVパスを渡す
    intraday_path = "docs/outputs/rbank9_intraday.csv"
    MP = {
        "RAW_TZ_INTRADAY": "Asia/Tokyo",
        "DISPLAY_TZ": "Asia/Tokyo",
    }

    intraday = read_any(intraday_path, MP["RAW_TZ_INTRADAY"], MP["DISPLAY_TZ"])

    if intraday.empty:
        print("⚠️ Intraday data is empty.")
        return

    # チャート出力
    plt.figure(figsize=(10, 5))
    plt.plot(intraday["time"], intraday["value"], color="red", linewidth=2)
    plt.title("R-BANK9 (1d)", color="lightcoral", fontsize=14)
    plt.xlabel("Time")
    plt.ylabel("Index Value")
    plt.grid(True, alpha=0.3)
    plt.savefig("docs/outputs/rbank9_1d.png", facecolor="#0E1117")

    print("✅ Chart generated successfully: docs/outputs/rbank9_1d.png")


if __name__ == "__main__":
    main()
