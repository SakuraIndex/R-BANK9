#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
intraday.csv から「終値レベル」を推定する保険スクリプト（強化版）。

想定する入力（docs/outputs/{index_key}_intraday.csv）:
- 正常:   ts,pct(またはvalue/levelなど) で複数行
- 変則:   列名ゆらぎ (ts/time/date/datetime, val/value/index/level/pct 等)
- 壊れ:   銘柄ヘッダ 1 行のみ（例: 5830.T,# いよぎん..., 5831.T, ...）
- 空/欠損: ファイルなし、または有効行が 0

挙動:
1) intraday から「当日分の最終値」を推定して返す
2) 取れなければ history の最新値 (docs/outputs/{index_key}_history.csv) にフォールバック
3) それも無理なら None
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Iterable

import pandas as pd


# ====== ユーティリティ ======
_DEF_TZ = "Asia/Tokyo"


def _is_header_only_intraday(text: str) -> bool:
    """
    壊れた intraday（銘柄ヘッダだけ 1 行）の判定。
    .T を含むティッカーや # コメントが多数並ぶパターンを検知。
    """
    if not text:
        return False
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) != 1:
        return False
    # 1 行しかなく、ティッカー拡張子や # を多く含むならヘッダ扱い
    line = lines[0]
    hit = sum(tok in line for tok in [".T", "#", "銘柄", "グループ"])
    # カンマが多く、数字より文字が多いケースも
    commas = line.count(",")
    digits = sum(ch.isdigit() for ch in line)
    return hit >= 1 and commas >= 3 and digits < commas  # かなり緩いが誤検知は少ない


def _read_csv_robust(csv_path: Path) -> pd.DataFrame:
    """
    多少壊れていても DataFrame に起こす。
    - BOM / 前後空白を除去
    - 1 行ヘッダのみ(銘柄一覧)は空 DataFrame を返す
    """
    if not csv_path.exists():
        return pd.DataFrame()

    txt = csv_path.read_text(encoding="utf-8", errors="ignore").replace("\ufeff", "").strip()
    if not txt:
        return pd.DataFrame()

    # 銘柄ヘッダ 1 行だけなら空扱い
    if _is_header_only_intraday(txt):
        return pd.DataFrame()

    # 通常読み
    try:
        df = pd.read_csv(pd.compat.StringIO(txt))
    except Exception:
        # どうしてもダメなら pandas に丸投げ
        df = pd.read_csv(csv_path)

    # 前後空白除去
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _best_datetime_col(df: pd.DataFrame) -> Optional[str]:
    hints = ["ts", "time", "timestamp", "date", "datetime"]
    low = [str(c).lower().strip() for c in df.columns]
    for h in hints:
        if h in low:
            c = df.columns[low.index(h)]
            s = pd.to_datetime(df[c], errors="coerce", utc=False)
            if s.notna().sum() >= max(3, len(s) // 5):
                df[c] = s
                return str(c)
    # 最大ヒット列（最後の砦）
    best, ok = None, -1
    for c in df.columns:
        s = pd.to_datetime(df[c], errors="coerce", utc=False)
        hit = int(s.notna().sum())
        if hit > ok:
            ok, best = hit, str(c)
            df[c] = s
    return best


def _best_value_col(df: pd.DataFrame, tcol: str) -> Optional[str]:
    candidates = []
    # 候補名を優先（pct/value/level/index など）
    prefer: Iterable[str] = ("pct", "value", "val", "level", "index", "ratio")
    low_map = {str(c).lower(): str(c) for c in df.columns if str(c) != tcol}

    for name in prefer:
        if name in low_map:
            c = low_map[name]
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() >= max(3, len(s) // 5):
                df[c] = s
                return c

    # だめなら最も数値ヒットが多い列
    for c in df.columns:
        if str(c) == tcol:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= max(3, len(s) // 5):
            candidates.append((str(c), int(s.notna().sum())))
            df[c] = s
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0] if candidates else None


def _latest_today_or_all(d: pd.DataFrame) -> Optional[float]:
    if d.empty:
        return None
    # 当日分優先
    today = pd.Timestamp.now(tz=_DEF_TZ).date()
    dd = d[d["ts"].dt.date == today]
    if dd.empty:
        dd = d
    return float(dd.iloc[-1]["val"])


# ====== 公開関数 ======
def guess_close_level(index_key: str, out_dir: Path) -> Optional[float]:
    """
    intraday → history の順で「直近レベル」を返す。
    戻り値が None のときはどちらからも取得不能。
    """
    # 1) intraday
    intraday_csv = out_dir / f"{index_key}_intraday.csv"
    df = _read_csv_robust(intraday_csv)
    if not df.empty and df.shape[1] >= 2:
        tcol = _best_datetime_col(df.copy())
        if tcol:
            vcol = _best_value_col(df.copy(), tcol)
            if vcol:
                d = pd.DataFrame(
                    {
                        "ts": pd.to_datetime(df[tcol], errors="coerce", utc=False),
                        "val": pd.to_numeric(df[vcol], errors="coerce"),
                    }
                ).dropna()
                d = d.sort_values("ts")
                v = _latest_today_or_all(d)
                if v is not None:
                    return v

    # 2) history フォールバック（date,value の 2 列想定／柔軟に解釈）
    hist_csv = out_dir / f"{index_key}_history.csv"
    if hist_csv.exists():
        try:
            h = pd.read_csv(hist_csv)
            # 列名ゆらぎ吸収
            cols = {c.lower().strip(): c for c in h.columns}
            dcol = None
            for k in ("date", "ts", "time", "timestamp"):
                if k in cols:
                    dcol = cols[k]
                    break
            vcol = None
            for k in ("value", "val", "level", "index", "pct", "ratio"):
                if k in cols:
                    vcol = cols[k]
                    break
            if dcol is None or vcol is None:
                # 2 列しかない場合は右列を値とみなす
                if h.shape[1] >= 2:
                    dcol = h.columns[0]
                    vcol = h.columns[1]
            if dcol is not None and vcol is not None:
                hh = pd.DataFrame(
                    {
                        "ts": pd.to_datetime(h[dcol], errors="coerce", utc=False).dt.tz_localize(None),
                        "val": pd.to_numeric(h[vcol], errors="coerce"),
                    }
                ).dropna()
                if not hh.empty:
                    hh = hh.sort_values("ts")
                    return float(hh.iloc[-1]["val"])
        except Exception:
            pass

    # 3) どうしても無理
    return None


# 直接叩くときの簡易テスト
if __name__ == "__main__":
    import sys

    key = sys.argv[1] if len(sys.argv) > 1 else "rbank9"
    out = Path("docs/outputs")
    v = guess_close_level(key, out)
    print(f"[level_from_intraday] key={key} -> {v}")
