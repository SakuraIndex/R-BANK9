import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

OUTPUT_DIR = "docs/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

INDEX_KEY = "ain10"  # ← リポジトリごとに変更（例: scoin_plus, rbank9, etc）
INDEX_TITLE = "AIN10"

# データ取得関数（例: 履歴CSVから）
def load_data():
    csv_path = os.path.join(OUTPUT_DIR, f"{INDEX_KEY}_history.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]
    if "date" not in df.columns:
        raise KeyError("CSV must contain 'date' column")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # volume列がない場合は仮の列を追加
    if "volume" not in df.columns:
        df["volume"] = 0.0

    return df


# チャート描画関数
def plot_chart(df, period_days: int, filename: str):
    if df.empty:
        print(f"[WARN] No data to plot for {filename}")
        return

    df_period = df[df["date"] >= (datetime.now() - timedelta(days=period_days))]

    # 移動平均線
    for w in [5, 25, 75]:
        df_period[f"SMA{w}"] = df_period["close"].rolling(w).mean()

    fig, ax1 = plt.subplots(figsize=(10, 5), facecolor="black")
    ax1.set_facecolor("black")

    # メイン線
    ax1.plot(df_period["date"], df_period["close"], color="pink", label="Index")
    ax1.plot(df_period["date"], df_period["SMA5"], color="skyblue", label="SMA5", linewidth=0.8)
    ax1.plot(df_period["date"], df_period["SMA25"], color="gold", label="SMA25", linewidth=0.8)
    ax1.plot(df_period["date"], df_period["SMA75"], color="lightgreen", label="SMA75", linewidth=0.8)

    ax1.set_xlabel("Date", color="white")
    ax1.set_ylabel("Index Value", color="white")
    ax1.tick_params(colors="white")
    ax1.legend(facecolor="black", labelcolor="white", loc="upper left")

    # 出来高（副軸）
    ax2 = ax1.twinx()
    ax2.bar(df_period["date"], df_period["volume"], alpha=0.3, color="white", label="Volume")
    ax2.set_ylabel("Volume", color="white")
    ax2.tick_params(colors="white")

    plt.title(f"{INDEX_TITLE} ({period_days}d)", color="pink")
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(out_path, facecolor="black")
    plt.close()
    print(f"✅ Saved: {out_path}")


def main():
    df = load_data()
    periods = {
        "1d": 1,
        "7d": 7,
        "1m": 30,
        "1y": 365,
    }

    for label, days in periods.items():
        filename = f"{INDEX_KEY}_{label}.png"
        plot_chart(df, days, filename)


if __name__ == "__main__":
    main()
