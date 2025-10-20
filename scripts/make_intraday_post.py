#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---- matplotlib (黒ベース) ----
plt.rcParams["figure.figsize"] = (14, 6)
plt.rcParams["axes.facecolor"] = "#0e0e10"
plt.rcParams["figure.facecolor"] = "#0e0e10"
plt.rcParams["axes.edgecolor"] = "#aaaaaa"
plt.rcParams["axes.labelcolor"] = "#cccccc"
plt.rcParams["xtick.color"] = "#cccccc"
plt.rcParams["ytick.color"] = "#cccccc"
plt.rcParams["text.color"] = "#cccccc"
LINE_COLOR = "#05f7f2"

JST = "Asia/Tokyo"

def to_jst_index(raw: pd.DataFrame, dt_col: str) -> pd.DataFrame:
    if dt_col not in raw.columns:
        raise ValueError(f"CSVに対象列 '{dt_col}' がありません。 列={list(raw.columns)}")
    dt = pd.to_datetime(raw[dt_col], utc=True, errors="coerce")
    if dt.isna().all():
        # UTCでなければ naive として解釈→UTC→JST に寄せる
        dt = pd.to_datetime(raw[dt_col], errors="coerce").dt.tz_localize("UTC")
    idx = dt.dt.tz_convert("Asia/Tokyo")
    out = raw.copy()
    out.index = idx
    return out.drop(columns=[dt_col])

def between_session(df_jst: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    # 09:00〜15:30 のような「時刻だけ」を受け取り、同日の範囲のみにフィルタ
    mask = (df_jst.index.time >= pd.Timestamp(start).time()) & \
           (df_jst.index.time <= pd.Timestamp(end).time())
    return df_jst.loc[mask]

def compute_series(df: pd.DataFrame, index_key: str, value_type: str, basis: str) -> pd.Series:
    """
    index_key: 使う列名（CSV中の列と一致していること）
    value_type:
      - "raw"     : 値そのまま
      - "percent" : 基準時刻(ラベル)からの%変化
      - "ratio"   : 前日終値比（= (値 / 前日終値 - 1)*100 ）
    basis:
      - "open@HH:MM" など → percentのときに使うラベル表記だけ
      - "prev_close"  → ratioのときの基準表記
    """
    if index_key not in df.columns:
        raise ValueError(f"CSV に対象列 '{index_key}' がありません。 列={list(df.columns)}")

    s = pd.to_numeric(df[index_key], errors="coerce")

    if value_type == "raw":
        return s

    if value_type == "percent":
        # 最初の有効値を基準にする（anchorの行をピンポイントで取れないケースがあるため）
        anchor = s.dropna().iloc[0]
        return (s / anchor - 1.0) * 100.0

    if value_type == "ratio":
        # 前日終値は CSV に 'prev_close' 列がある前提 or 先頭値を近似基準に fallback
        if "prev_close" in df.columns:
            prev = pd.to_numeric(df["prev_close"], errors="coerce").dropna()
            prev_val = prev.iloc[0] if not prev.empty else np.nan
        else:
            prev_val = np.nan
        if np.isnan(prev_val):
            # フォールバック：午前の最初の有効値を「前日終値に近い基準」として扱う
            prev_val = s.dropna().iloc[0]
        return (s / prev_val - 1.0) * 100.0

    raise ValueError(f"Unknown value_type: {value_type}")

def snapshot_title(label: str) -> str:
    now_jst = pd.Timestamp.utcnow().tz_localize("UTC").tz_convert(JST)
    return f"{label} Intraday Snapshot ({now_jst:%Y/%m/%d %H:%M})"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-key", required=True, help="CSVの対象列名（例: ASTRA4）")
    ap.add_argument("--csv", required=True, help="入力CSV")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-text", required=True)
    ap.add_argument("--snapshot-png", required=True)
    ap.add_argument("--session-start", required=True)  # "09:00"
    ap.add_argument("--session-end", required=True)    # "15:30"
    ap.add_argument("--day-anchor", required=True)     # 表示ラベル用
    ap.add_argument("--basis", required=True)          # "open@09:00" / "prev_close"
    ap.add_argument("--label", default=None)           # 図の凡例/表題に使うラベル
    ap.add_argument("--dt-col", default="Datetime")    # 日時列名（デフォルト Datetime）
    ap.add_argument("--value-type", choices=["raw", "percent", "ratio"], default="percent")

    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    raw = pd.read_csv(csv_path)
    df_jst = to_jst_index(raw, args.dt_col)
    df_sess = between_session(df_jst, args.session_start, args.session_end)

    if df_sess.empty:
        raise ValueError("セッション内データがありません。")

    s = compute_series(df_sess, args.index_key, args.value_type, args.basis)
    last_pct = float(s.dropna().iloc[-1]) if not s.dropna().empty else float("nan")

    # --- PNG: プロット ---
    fig, ax = plt.subplots()
    (s).plot(ax=ax, color=LINE_COLOR, label=args.label or args.index_key)
    ax.set_title(snapshot_title(args.label or args.index_key))
    ylabel = "Change vs Prev Close (%)" if args.value_type == "ratio" else "Change vs Anchor (%)"
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time")
    ax.legend(loc="upper left")
    fig.tight_layout()
    Path(args.snapshot_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.snapshot_png, dpi=150)
    plt.close(fig)

    # --- POSTテキスト ---
    sign = "▲" if last_pct >= 0 else "▼"
    basis_label = args.basis
    post_lines = [
        f"{sign} {args.label or args.index_key} 日中スナップショット（{pd.Timestamp.utcnow().tz_localized('UTC').tz_convert(JST):%Y/%m/%d %H:%M}）",
        f"{last_pct:+.2f}%（基準: {basis_label}）",
        f"#{(args.label or args.index_key).upper()} #日本株",
    ]
    Path(args.out_text).write_text("\n".join(post_lines), encoding="utf-8")

    # --- JSON（ダッシュボード等で使用） ---
    stats = {
        "index_key": (args.label or args.index_key).upper(),
        "label": (args.label or args.index_key).upper(),
        "pct_intraday": last_pct,
        "basis": args.basis,
        "session": {"start": args.session_start, "end": args.session_end, "anchor": args.day_anchor},
        "updated_at": f"{pd.Timestamp.utcnow().tz_localize('UTC').tz_convert(JST):%Y-%m-%dT%H:%M:%S%z}",
    }
    Path(args.out_json).write_text(pd.Series(stats).to_json(force_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---- matplotlib (黒ベース) ----
plt.rcParams["figure.figsize"] = (14, 6)
plt.rcParams["axes.facecolor"] = "#0e0e10"
plt.rcParams["figure.facecolor"] = "#0e0e10"
plt.rcParams["axes.edgecolor"] = "#aaaaaa"
plt.rcParams["axes.labelcolor"] = "#cccccc"
plt.rcParams["xtick.color"] = "#cccccc"
plt.rcParams["ytick.color"] = "#cccccc"
plt.rcParams["text.color"] = "#cccccc"
LINE_COLOR = "#05f7f2"

JST = "Asia/Tokyo"

def to_jst_index(raw: pd.DataFrame, dt_col: str) -> pd.DataFrame:
    if dt_col not in raw.columns:
        raise ValueError(f"CSVに対象列 '{dt_col}' がありません。 列={list(raw.columns)}")
    dt = pd.to_datetime(raw[dt_col], utc=True, errors="coerce")
    if dt.isna().all():
        # UTCでなければ naive として解釈→UTC→JST に寄せる
        dt = pd.to_datetime(raw[dt_col], errors="coerce").dt.tz_localize("UTC")
    idx = dt.dt.tz_convert("Asia/Tokyo")
    out = raw.copy()
    out.index = idx
    return out.drop(columns=[dt_col])

def between_session(df_jst: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    # 09:00〜15:30 のような「時刻だけ」を受け取り、同日の範囲のみにフィルタ
    mask = (df_jst.index.time >= pd.Timestamp(start).time()) & \
           (df_jst.index.time <= pd.Timestamp(end).time())
    return df_jst.loc[mask]

def compute_series(df: pd.DataFrame, index_key: str, value_type: str, basis: str) -> pd.Series:
    """
    index_key: 使う列名（CSV中の列と一致していること）
    value_type:
      - "raw"     : 値そのまま
      - "percent" : 基準時刻(ラベル)からの%変化
      - "ratio"   : 前日終値比（= (値 / 前日終値 - 1)*100 ）
    basis:
      - "open@HH:MM" など → percentのときに使うラベル表記だけ
      - "prev_close"  → ratioのときの基準表記
    """
    if index_key not in df.columns:
        raise ValueError(f"CSV に対象列 '{index_key}' がありません。 列={list(df.columns)}")

    s = pd.to_numeric(df[index_key], errors="coerce")

    if value_type == "raw":
        return s

    if value_type == "percent":
        # 最初の有効値を基準にする（anchorの行をピンポイントで取れないケースがあるため）
        anchor = s.dropna().iloc[0]
        return (s / anchor - 1.0) * 100.0

    if value_type == "ratio":
        # 前日終値は CSV に 'prev_close' 列がある前提 or 先頭値を近似基準に fallback
        if "prev_close" in df.columns:
            prev = pd.to_numeric(df["prev_close"], errors="coerce").dropna()
            prev_val = prev.iloc[0] if not prev.empty else np.nan
        else:
            prev_val = np.nan
        if np.isnan(prev_val):
            # フォールバック：午前の最初の有効値を「前日終値に近い基準」として扱う
            prev_val = s.dropna().iloc[0]
        return (s / prev_val - 1.0) * 100.0

    raise ValueError(f"Unknown value_type: {value_type}")

def snapshot_title(label: str) -> str:
    now_jst = pd.Timestamp.utcnow().tz_localize("UTC").tz_convert(JST)
    return f"{label} Intraday Snapshot ({now_jst:%Y/%m/%d %H:%M})"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-key", required=True, help="CSVの対象列名（例: ASTRA4）")
    ap.add_argument("--csv", required=True, help="入力CSV")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-text", required=True)
    ap.add_argument("--snapshot-png", required=True)
    ap.add_argument("--session-start", required=True)  # "09:00"
    ap.add_argument("--session-end", required=True)    # "15:30"
    ap.add_argument("--day-anchor", required=True)     # 表示ラベル用
    ap.add_argument("--basis", required=True)          # "open@09:00" / "prev_close"
    ap.add_argument("--label", default=None)           # 図の凡例/表題に使うラベル
    ap.add_argument("--dt-col", default="Datetime")    # 日時列名（デフォルト Datetime）
    ap.add_argument("--value-type", choices=["raw", "percent", "ratio"], default="percent")

    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    raw = pd.read_csv(csv_path)
    df_jst = to_jst_index(raw, args.dt_col)
    df_sess = between_session(df_jst, args.session_start, args.session_end)

    if df_sess.empty:
        raise ValueError("セッション内データがありません。")

    s = compute_series(df_sess, args.index_key, args.value_type, args.basis)
    last_pct = float(s.dropna().iloc[-1]) if not s.dropna().empty else float("nan")

    # --- PNG: プロット ---
    fig, ax = plt.subplots()
    (s).plot(ax=ax, color=LINE_COLOR, label=args.label or args.index_key)
    ax.set_title(snapshot_title(args.label or args.index_key))
    ylabel = "Change vs Prev Close (%)" if args.value_type == "ratio" else "Change vs Anchor (%)"
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time")
    ax.legend(loc="upper left")
    fig.tight_layout()
    Path(args.snapshot_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.snapshot_png, dpi=150)
    plt.close(fig)

    # --- POSTテキスト ---
    sign = "▲" if last_pct >= 0 else "▼"
    basis_label = args.basis
    post_lines = [
        f"{sign} {args.label or args.index_key} 日中スナップショット（{pd.Timestamp.utcnow().tz_localized('UTC').tz_convert(JST):%Y/%m/%d %H:%M}）",
        f"{last_pct:+.2f}%（基準: {basis_label}）",
        f"#{(args.label or args.index_key).upper()} #日本株",
    ]
    Path(args.out_text).write_text("\n".join(post_lines), encoding="utf-8")

    # --- JSON（ダッシュボード等で使用） ---
    stats = {
        "index_key": (args.label or args.index_key).upper(),
        "label": (args.label or args.index_key).upper(),
        "pct_intraday": last_pct,
        "basis": args.basis,
        "session": {"start": args.session_start, "end": args.session_end, "anchor": args.day_anchor},
        "updated_at": f"{pd.Timestamp.utcnow().tz_localize('UTC').tz_convert(JST):%Y-%m-%dT%H:%M:%S%z}",
    }
    Path(args.out_json).write_text(pd.Series(stats).to_json(force_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
