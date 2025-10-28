#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
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
CLAMP_PCT = 15.0   # 1d 用の軟クランプ

# ====== theme ======
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

# ====== CSV utilities ======
PCT_CANDIDATES   = ("pct","rtn","change_pct","change%","intraday_pct","pct_intraday","rtn_intraday")
LEVEL_CANDIDATES = ("level","index","mean","value","val", INDEX_KEY, INDEX_KEY.replace("-","").replace("_",""))

def _parse_dt(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        m = float(pd.to_numeric(s, errors="coerce").dropna().median()) if s.notna().any() else 0.0
        if m > 1e12:   # ms
            return pd.to_datetime(s, unit="ms", errors="coerce")
        elif m > 1e9:  # s
            return pd.to_datetime(s, unit="s", errors="coerce")
        else:
            return pd.to_datetime(s, errors="coerce")
    return pd.to_datetime(s.astype(str), errors="coerce")

def _choose_ts_col(df: pd.DataFrame) -> str | None:
    # もっとも datetime に変換できる列
    best, best_ok = None, -1
    for c in df.columns:
        s = _parse_dt(df[c])
        ok = s.notna().sum()
        if ok > best_ok:
            best, best_ok = c, ok
    return best

def _choose_value_col(df: pd.DataFrame, ts_name: str, prefer_pct: bool) -> tuple[str, str] | tuple[None, None]:
    """ts 列を必ず除外して値列を返す。kind は 'pct' or 'level'。"""
    cols_lower = {c: c.lower().strip() for c in df.columns if c != ts_name}

    # pct を優先採用（必要なとき）
    if prefer_pct:
        for k in PCT_CANDIDATES:
            for c, lc in cols_lower.items():
                if lc == k:
                    s = pd.to_numeric(df[c], errors="coerce")
                    if s.notna().sum() > 0:
                        return c, "pct"

    # level 候補名
    for k in LEVEL_CANDIDATES:
        for c, lc in cols_lower.items():
            if lc == k:
                s = pd.to_numeric(df[c], errors="coerce")
                if s.notna().sum() > 0:
                    return c, "level"

    # 数値の有効値が最も多い列
    best, mx = None, -1
    for c in df.columns:
        if c == ts_name: 
            continue
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
    ts_col = _choose_ts_col(df)
    if not ts_col:
        return pd.DataFrame()
    out = pd.DataFrame({"ts": _parse_dt(df[ts_col])})
    for c in df.columns:
        if c == ts_col: 
            continue
        out[c] = pd.to_numeric(df[c], errors="coerce")
    out = out.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    mask = (out["ts"].dt.year >= 2000) & (out["ts"].dt.year <= 2100)
    return out.loc[mask].reset_index(drop=True)

# ====== % helpers ======
def _stable_baseline(df_day: pd.DataFrame) -> float | None:
    # df_day: 必須列 ts, val
    if df_day.empty or "ts" not in df_day or "val" not in df_day:
        return None
    after10 = (df_day["ts"].dt.hour > 10) | ((df_day["ts"].dt.hour == 10) & (df_day["ts"].dt.minute >= 0))
    cand = df_day.loc[after10 & df_day["val"].notna()]
    if not cand.empty:
        return float(cand.iloc[0]["val"])
    s = df_day["val"].dropna()
    return float(s.iloc[0]) if not s.empty else None

def _calc_pct(base: float, v: float) -> float:
    denom = max(abs(base), EPS)
    p = (v - base) / denom * 100.0
    if p >  CLAMP_PCT: return CLAMP_PCT
    if p < -CLAMP_PCT: return -CLAMP_PCT
    return p

# ====== loaders ======
def load_for_1d() -> pd.DataFrame | None:
    """intraday から % 系列を返す。pct 列があればそれを採用、無ければ level→% 計算。"""
    for name in (f"{INDEX_KEY}_intraday.csv", f"{INDEX_KEY}_1d.csv"):
        p = OUT_DIR / name
        df = _load_csv(p)
        if df.empty:
            continue

        ts_name = "ts"
        val_col, kind = _choose_value_col(df, ts_name=ts_name, prefer_pct=True)
        if not val_col:
            continue

        if kind == "pct":
            out = df[[ts_name, val_col]].rename(columns={val_col: "pct"}).dropna()
            out["pct"] = out["pct"].clip(-CLAMP_PCT, CLAMP_PCT)
            return out

        # level → %
        day = df[ts_name].dt.floor("D").iloc[-1]
        d = df[df[ts_name].dt.floor("D") == day][[ts_name, val_col]].dropna().reset_index(drop=True)
        if d.empty:
            continue
        d = d.rename(columns={val_col: "val"})
        base = _stable_baseline(d)
        if base is None:
            continue
        d["pct"] = d["val"].apply(lambda v: _calc_pct(base, v))
        return d[[ts_name, "pct"]]
    return None

def load_for_level_span(days: int) -> pd.DataFrame | None:
    """7d/1m/1y 用レベル系列。history を最優先。"""
    df = _load_csv(OUT_DIR / f"{INDEX_KEY}_history.csv")
    if df.empty:
        alt = _load_csv(OUT_DIR / f"{INDEX_KEY}_{days}d.csv")  # 例: 7d.csv があれば使用
        if alt.empty:
            return None
        df = alt

    ts_name = "ts"
    lvl_col, _ = _choose_value_col(df, ts_name=ts_name, prefer_pct=False)
    if not lvl_col:
        return None

    last = df[ts_name].max()
    span = df[df[ts_name] >= (last - pd.Timedelta(days=days))][[ts_name, lvl_col]].dropna()
    if span.empty:
        return None
    return span.rename(columns={lvl_col: "level"}).reset_index(drop=True)

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
