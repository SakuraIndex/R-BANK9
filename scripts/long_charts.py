# scripts/long_charts.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re
import pandas as pd
import pytz
import matplotlib
import matplotlib.pyplot as plt

JP = pytz.timezone("Asia/Tokyo")
NY = pytz.timezone("America/New_York")

BG, FG, TITLE = "#0E1117", "#E6E6E6", "#f2b6c6"
UP, DOWN, GRID_A = "#3bd6c6", "#ff6b6b", 0.25

matplotlib.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": FG,
    "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
    "text.color": FG, "grid.color": FG, "savefig.facecolor": BG,
})

OUTDIR = os.path.join("docs", "outputs")

# -----------------------------
# Utility
# -----------------------------
def _lower(df): df.columns = [str(c).strip().lower() for c in df.columns]; return df

def _find_time_col(cols):
    for c in cols:
        if re.search(r"time|日時|date|timestamp|時刻", c): return c
    if cols: return cols[0]
    return None

def read_intraday(path:str)->pd.DataFrame:
    if not os.path.exists(path): return pd.DataFrame(columns=["time","value"])
    raw = pd.read_csv(path, dtype=str)
    if raw.empty: return pd.DataFrame(columns=["time","value"])
    df = _lower(raw)
    tcol = _find_time_col(df.columns)
    if tcol is None: return pd.DataFrame(columns=["time","value"])

    # pick numeric col
    vcol=None
    for c in df.columns:
        if c==tcol: continue
        if any(k in c for k in ["value","index","score","mean"]): vcol=c;break
    if vcol is None:
        # fallback: first numeric-looking col
        for c in df.columns:
            if c==tcol: continue
            try:
                pd.to_numeric(df[c])
                vcol=c;break
            except: pass

    # parse time
    t=pd.to_datetime(df[tcol], errors="coerce", utc=True)
    if t.dt.tz is None: t=pd.to_datetime(df[tcol], errors="coerce").dt.tz_localize("UTC")
    # detect JST-like pattern
    if "jst" in tcol.lower() or t.dt.hour.mean()>=0 and t.dt.hour.mean()<=23:
        t=t.dt.tz_convert(JP)
    out=pd.DataFrame({"time":t.dt.tz_convert(JP)})
    if vcol:
        out["value"]=pd.to_numeric(df[vcol], errors="coerce")
    else:
        out["value"]=pd.to_numeric(df.iloc[:,1], errors="coerce")
    out=out.dropna(subset=["time","value"]).sort_values("time")
    return out.reset_index(drop=True)

def resample(df, rule="1min"):
    if df.empty: return df
    tmp=df.set_index("time").sort_index()
    g=tmp[["value"]].resample(rule).mean()
    g["value"]=g["value"].interpolate(limit_direction="both")
    return g.reset_index()

def window(df,key):
    if df.empty: return df
    now_jst=pd.Timestamp.now(tz=JP)
    today=now_jst.normalize()
    if key in ("astra4","rbank9"):
        s=pd.Timestamp(f"{today.date()} 09:00",tz=JP)
        e=pd.Timestamp(f"{today.date()} 15:30",tz=JP)
        w=df[(df["time"]>=s)&(df["time"]<=e)]
    elif key=="ain10":
        tny=df["time"].dt.tz_convert(NY)
        day=pd.Timestamp.now(tz=NY).normalize()
        s=pd.Timestamp(f"{day.date()} 09:30",tz=NY)
        e=pd.Timestamp(f"{day.date()} 16:00",tz=NY)
        w=df[(tny>=s)&(tny<=e)]
    else:  # scoin_plus
        from_ts=pd.Timestamp.now(tz=JP)-pd.Timedelta(hours=24)
        w=df[df["time"]>=from_ts]
    if w.empty:
        w=df.tail(600)
    return w.reset_index(drop=True)

def robust_pct(vals):
    vals=pd.to_numeric(vals, errors="coerce").dropna().values
    if len(vals)<2: return None
    base, last=vals[0], vals[-1]
    if abs(base)<1e-9: return 0.0
    p=((last/base)-1)*100
    return p if abs(p)<=100 else 0.0

def decorate(ax,title,xl,yl):
    ax.set_title(title,color=TITLE,fontsize=20,pad=12)
    ax.set_xlabel(xl);ax.set_ylabel(yl);ax.grid(True,alpha=GRID_A)
    for s in ax.spines.values(): s.set_color(FG)

def save(fig,path): fig.savefig(path,facecolor=BG,bbox_inches="tight");plt.close(fig)

# -----------------------------
# Main
# -----------------------------
def main():
    os.makedirs(OUTDIR,exist_ok=True)
    key=os.environ.get("INDEX_KEY","index").strip().lower()
    name=key.upper().replace("_","")

    intraday_csv=os.path.join(OUTDIR,f"{key}_intraday.csv")
    history_csv =os.path.join(OUTDIR,f"{key}_history.csv")

    try:
        df=read_intraday(intraday_csv)
        df=window(df,key)
        df=resample(df)
    except Exception as e:
        print("load fail",e)
        df=pd.DataFrame(columns=["time","value"])

    delta=robust_pct(df["value"]) if not df.empty else None
    color=UP if (delta is None or delta>=0) else DOWN

    # 1d
    fig,ax=plt.subplots(figsize=(16,7),layout="constrained")
    decorate(ax,f"{name} (1d)","Time","Index Value")
    if not df.empty: ax.plot(df["time"],df["value"],lw=2.2,color=color)
    else: ax.text(0.5,0.5,"No data",transform=ax.transAxes,ha="center",va="center",alpha=0.6)
    save(fig,os.path.join(OUTDIR,f"{key}_1d.png"))

    # 7d/1m/1y from history
    if os.path.exists(history_csv):
        h=pd.read_csv(history_csv)
        if "date" in h and "value" in h:
            h["date"]=pd.to_datetime(h["date"],errors="coerce")
            h["value"]=pd.to_numeric(h["value"],errors="coerce")
            for days,label in [(7,"7d"),(30,"1m"),(365,"1y")]:
                fig,ax=plt.subplots(figsize=(16,7),layout="constrained")
                decorate(ax,f"{name} ({label})","Date","Index Value")
                hh=h.tail(days)
                if len(hh)>=2:
                    col=UP if hh["value"].iloc[-1]>=hh["value"].iloc[0] else DOWN
                    ax.plot(hh["date"],hh["value"],lw=2.0,color=col)
                else:
                    ax.text(0.5,0.5,"No data",transform=ax.transAxes,ha="center",va="center",alpha=0.5)
                save(fig,os.path.join(OUTDIR,f"{key}_{label}.png"))

    # write delta txt
    txt=f"{name} 1d: {(0.0 if delta is None else delta):+0.2f}%"
    with open(os.path.join(OUTDIR,f"{key}_post_intraday.txt"),"w",encoding="utf-8") as f:
        f.write(txt)

if __name__=="__main__":
    main()
