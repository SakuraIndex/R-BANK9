import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timezone

OUTDIR = "docs/outputs"

def read_intraday(csv_path: str):
    """CSVを読み込み、R_BANK9列があればそれを使用"""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    
    # Datetime列の検出
    time_col = next((c for c in df.columns if "time" in c.lower() or "datetime" in c.lower()), df.columns[0])
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    # R_BANK9列を優先して使用
    if "R_BANK9" in df.columns:
        df = df[[time_col, "R_BANK9"]].rename(columns={time_col: "time", "R_BANK9": "value"})
    elif "value" in df.columns:
        df = df[[time_col, "value"]].rename(columns={time_col: "time"})
    else:
        # R_BANK9列がない場合は数値列の平均を取る
        numeric_cols = df.select_dtypes(include=["number"]).columns
        df["value"] = df[numeric_cols].mean(axis=1)
        df = df[[time_col, "value"]].rename(columns={time_col: "time"})
    
    # 時間順に並べ替え
    df = df.sort_values("time").dropna()
    return df


def plot_chart(df: pd.DataFrame, title: str, outpath: str):
    """1日チャートを描画"""
    plt.figure(figsize=(8, 3))
    plt.plot(df["time"], df["value"], color="cyan" if df["value"].iloc[-1] >= df["value"].iloc[0] else "red", linewidth=1.4)
    plt.title(title, fontsize=10)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    key = "rbank9"
    csv_path = os.path.join(OUTDIR, f"{key}_intraday.csv")
    png_path = os.path.join(OUTDIR, f"{key}_intraday.png")

    df = read_intraday(csv_path)
    if len(df) == 0:
        print("❌ データがありません。CSVを確認してください。")
        return

    plot_chart(df, "RBANK9 (1d)", png_path)
    print(f"✅ チャート生成完了: {png_path}")

    # 騰落率を算出
    start_value = df["value"].iloc[0]
    end_value = df["value"].iloc[-1]
    change = ((end_value - start_value) / abs(start_value)) * 100 if start_value != 0 else 0

    print(f"📊 騰落率: {change:+.2f}%")


if __name__ == "__main__":
    main()
