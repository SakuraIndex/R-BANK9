#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.dates import AutoDateLocator, AutoDateFormatter

# ====== basic ======
INDEX_KEY = os.environ.get("INDEX_KEY", "rbank9")
OUT_DIR   = Path(os.environ.get("OUT_DIR", "docs/outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# % 安全側
EPS       = 1e-6
CLAMP_PCT = 15.0   # 1d の描画を落ち着かせるための緩クランプ

# ====== helpers ======
def _apply_dark(fig, ax):
    fig.set_size_inches(16, 8)
    fig.set_dpi(180)
    fig.patch.set_facecolor("#111317")
    ax.set_facecolor("#111317")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(axis="both", colors="#ffffff", labelsize=10)
    ax.yaxis.label.set_color("#ffffff")
    ax.xaxis.label.set_color("#ffffff")
    ax.title.set_color("#ffffff")
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.18, color="#ffffff")
    ax.grid(True, which="minor", linestyle="-", linewidth=0.10, alpha=0.10, color="#ffffff")

def _parse_dt(s: pd.Series) -> pd.Series:
    """ISO/一般文字列→datetime。数値はユニックス秒/ミリ秒を推定。"""
    if pd.api.types.is_numeric_dtype(s):
        m = float(s.dropna().astype(float).median()) if s.notna().any() else 0.0
        if m > 1e12:   # ms
            return pd.to_datetime(s, unit="ms", errors="coerce")
        elif m > 1e9:  # s
            return pd.to_datetime(s, unit="s", errors="coerce")
        else:
            # yyyymmdd などの整数もここに来るので to_datetime 任せ
            return pd.to_datetime(s.astype("Int64"), errors="coerce")
    return pd.to_datetime(s.astype(str), errors="coerce")

PCT_CANDIDATES   = ("pct","rtn","change_pct","change%","intraday_pct","pct_intraday","rtn_intraday")
LEVEL_CANDIDATES = ("level","index","mean","value","val", INDEX_KEY, INDEX_KEY.replace("-","").replace("_",""))

def _choose_ts_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        s = _parse_dt(df[c])
        if s.notna().sum() >= max(5, len(s)//4):
            df[c] = s
            return c
    # fallback: 先頭列を時刻化トライ
    c = df.columns[0]
    df[c] = _parse_dt(df[c])
    return c if df[c].notna().any() else None

def _choose_value_col(df: pd.DataFrame, prefer_pct=False) -> tuple[str, str] | tuple[None, None]:
    cols_lower = {c: c.lower().strip() for c in df.columns}
    # 明示優先
    if prefer_pct:
        for k in PCT_CANDIDATES:
            for c, lc in cols_lower.items():
                if k == lc:
                    if pd.to_numeric(df[c], errors="coerce").notna().sum() > 0:
                        return c, "pct"
    # level候補
    for k in LEVEL_CANDIDATES:
        for c, lc in cols_lower.items():
            if k == lc:
                if pd.to_numeric(df[c], errors="coerce").notna().sum() > 0:
                    return c, "level"
    # 数値列の最多有効値
    best, kind = None, None
    mx = -1
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        cnt = s.notna().sum()
        if cnt > mx:
            mx, best = cnt, c
    if best:
        return best, "level"
    return None, None

def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty or df.shape[1] < 2:
        return pd.DataFrame()
    ts_col = _choose_ts_col(df.copy())
    if not ts_col:
        return pd.DataFrame()
    out = pd.DataFrame({"ts": _parse_dt(df[ts_col])})
    # 全数値列も持っておく（後で選ぶ）
    for c in df.columns:
        if c == ts_col: continue
        out[c] = pd.to_numeric(df[c], errors="coerce")
    out = out.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    # 年のレンジ sanity
    mask = (out["ts"].dt.year >= 2000) & (out["ts"].dt.year <= 2100)
    return out.loc[mask].reset_index(drop=True)

def _stable_baseline(df_day: pd.DataFrame) -> float | None:
    if df_day.empty: return None
    # 10:00以降優先（東京を想定、米市場でもAM帯が先行して安定しやすい）
    after10 = (df_day["ts"].dt.hour > 10) | ((df_day["ts"].dt.hour == 10) & (df_day["ts"].dt.minute >= 0))
    cand = df_day.loc[after10]
    s = pd.to_numeric(cand.iloc[:, 1], errors="coerce") if cand.shape[1] >= 2 else pd.Series(dtype=float)
    if not s.empty and s.notna().any():
        return float(s.dropna().iloc[0])
    # フォールバック：最初の有効値
    for c in df_day.columns:
        if c == "ts": continue
        s = pd.to_numeric(df_day[c], errors="coerce")
        if s.notna().any():
            return float(s.dropna().iloc[0])
    return None

def _calc_pct(base: float, v: float) -> float:
    denom = max(abs(base), EPS)
    p = (v - base) / denom * 100.0
    # 軟クランプ（過大な印象を抑える）
    if p >  CLAMP_PCT: return CLAMP_PCT
    if p < -CLAMP_PCT: return -CLAMP_PCT
    return p

# ====== loaders for spans ======
def load_for_1d() -> pd.DataFrame | None:
    """intraday から % 系列を返す。pct 列があればそれを採用、無ければ level→% 計算。"""
    # 優先: *_intraday.csv / 次: *_1d.csv
    for name in (f"{INDEX_KEY}_intraday.csv", f"{INDEX_KEY}_1d.csv"):
        p = OUT_DIR / name
        df = _load_csv(p)
        if df.empty: 
            continue

        # pct を優先採用
        pct_col, kind = _choose_value_col(df.copy(), prefer_pct=True)
        if pct_col and kind == "pct":
            out = df[["ts", pct_col]].rename(columns={pct_col: "pct"}).dropna()
            # 変な巨大値は丸める（念のため）
            out["pct"] = out["pct"].clip(-CLAMP_PCT, CLAMP_PCT)
            return out

        # level → % 計算
        lvl_col, _ = _choose_value_col(df.copy(), prefer_pct=False)
        if not lvl_col:
            continue
        day = df["ts"].dt.floor("D").iloc[-1]
        d = df[df["ts"].dt.floor("D") == day][["ts", lvl_col]].dropna().reset_index(drop=True)
        if d.empty:
            continue
        base = _stable_baseline(d.rename(columns={lvl_col: "val"}))
        if base is None:
            continue
        d["pct"] = d[lvl_col].apply(lambda v: _calc_pct(base, v))
        return d[["ts", "pct"]]
    return None

def load_for_level_span(days: int) -> pd.DataFrame | None:
    """7d/1m/1y 用レベル系列。history を最優先。"""
    # 優先: *_history.csv
    hist = _load_csv(OUT_DIR / f"{INDEX_KEY}_history.csv")
    if hist.empty:
        # 期間専用CSVがあれば使用
        alt = _load_csv(OUT_DIR / f"{INDEX_KEY}_{days}d.csv")  # 例: 7d
        if alt.empty:
            return None
        df = alt
    else:
        df = hist

    # レベル列を選択
    lvl_col, _ = _choose_value_col(df.copy(), prefer_pct=False)
    if not lvl_col:
        return None

    last = df["ts"].max()
    span = df[df["ts"] >= (last - pd.Timedelta(days=days))][["ts", lvl_col]].dropna()
    if span.empty:
        return None
    span = span.rename(columns={lvl_col: "level"}).reset_index(drop=True)
    return span

# ====== plotting ======
def plot_series(df: pd.DataFrame, ycol: str, title: str, ylabel: str, out_png: Path):
    fig, ax = plt.subplots()
    _apply_dark(fig, ax)
    ax.set_title(title, fontsize=26, fontweight="bold", pad=18)
    ax.set_xlabel("Time", labelpad=10)
    ax.set_ylabel(ylabel, labelpad=10)

    ax.plot(df["ts"].values, df[ycol].values, linewidth=2.6)

    major = AutoDateLocator(minticks=5, maxticks=10)
    ax.xaxis.set_major_locator(major)
    ax.xaxis.set_major_formatter(AutoDateFormatter(major))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.yaxis.set_minor_locator(MaxNLocator(nbins=50))

    fig.tight_layout()
    fig.savefig(out_png, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"[long] wrote {out_png}")

# ====== main ======
def main():
    # 1d: %
    d1 = load_for_1d()
    if d1 is not None and not d1.empty:
        plot_series(d1, "pct", f"{INDEX_KEY.upper()} (1d)", "Change (%)", OUT_DIR / f"{INDEX_KEY}_1d.png")
    else:
        print("[long] 1d: no series")

    # 7d / 1m(=30d) / 1y(=365d): level
    for span, days in [("7d", 7), ("1m", 30), ("1y", 365)]:
        d = load_for_level_span(days)
        if d is None or d.empty:
            print(f"[long] {span}: no series")
            continue
        plot_series(d, "level", f"{INDEX_KEY.upper()} ({span} level)", "Level (index)", OUT_DIR / f"{INDEX_KEY}_{span}.png")

if __name__ == "__main__":
    main()
