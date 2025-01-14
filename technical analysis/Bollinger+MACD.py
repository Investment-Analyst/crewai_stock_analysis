import matplotlib
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import sys

# 若需要顯示中文，可指定字型 (Windows 舉例)
# matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
# matplotlib.rcParams["axes.unicode_minus"] = False

# 1. 下載資料 (可換成你想要的 ticker)
start_date = "2020-01-01"
end_date   = datetime.datetime.today().strftime("%Y-%m-%d")

df = yf.download("USDCNY=X", start=start_date, end=end_date, interval="1d")
if df.empty:
    print("無法下載資料或資料為空")
    sys.exit()

df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

# 2. 計算布林通道 (±2 * STD)
df["MA20"] = df["Close"].rolling(20).mean()
df["MA10"] = df["Close"].rolling(10).mean()
df["STD20"] = df["Close"].rolling(20).std()
df["UpperBand"] = df["MA20"] + 1.5 * df["STD20"]
df["LowerBand"] = df["MA20"] - 1.5 * df["STD20"]

# 3. 計算 MACD
df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"] = df["EMA12"] - df["EMA26"]
df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

df.dropna(inplace=True)
if df.empty:
    print("技術指標計算後資料不足")
    sys.exit()

# 4. 建立 Signal 欄位
df["Signal"] = 0

# 5. 逐列 (row-by-row) 計算
for i in range(1, len(df)):
    date_str = df.index[i].strftime("%Y-%m-%d")

    # 取出當天所需變數
    high_price = df["High"].iloc[i]
    low_price = df["Low"].iloc[i]
    ma10 = df["MA10"].iloc[i]
    ma20 = df["MA20"].iloc[i]
    upper_band = df["UpperBand"].iloc[i]
    lower_band = df["LowerBand"].iloc[i]

    # MACD：前一天 vs 當天
    prev_macd    = df["MACD"].iloc[i - 1]
    prev_signal  = df["MACD_Signal"].iloc[i - 1]
    curr_macd    = df["MACD"].iloc[i]
    curr_signal  = df["MACD_Signal"].iloc[i]

    # MACD 死亡 & 黃金交叉
    cond_macd_dead_cross = (prev_macd > prev_signal) and (curr_macd < curr_signal)
    cond_macd_golden_cross = (prev_macd < prev_signal) and (curr_macd > curr_signal)

    # 判斷上軌 / 下軌
    cond_upper_band = (high_price >= upper_band) & (high_price > ma20) & cond_macd_dead_cross
    cond_lower_band = (low_price <= lower_band) & (low_price < ma20) & cond_macd_golden_cross

    if pd.Series(cond_upper_band).any():
        df.at[df.index[i], "Signal"] = -1
        print(f"{date_str} | 做空 -> 原因: 觸及上軌 + MACD 死亡交叉")
    elif pd.Series(cond_lower_band).any():
        df.at[df.index[i], "Signal"] = 1
        print(f"{date_str} | 做多 -> 原因: 觸及下軌 + MACD 黃金交叉")
    else:
        df.at[df.index[i], "Signal"] = 0

# 6. 轉為持倉 (Position)
df["Position"] = 0
df.loc[df.index[0], "Position"] = 0
for i in range(1, len(df)):
    if df["Signal"].iloc[i] == 0:
        df.at[df.index[i], "Position"] = df["Position"].iloc[i - 1]
    else:
        df.at[df.index[i], "Position"] = df["Signal"].iloc[i]

# 7. 計算簡易報酬
df["Daily_Return"] = df["Close"].pct_change()
df["Strategy_Return"] = df["Position"].shift(1) * df["Daily_Return"]
df["Strategy_Cum"] = (1 + df["Strategy_Return"]).cumprod()
df["BuyHold_Cum"] = (1 + df["Daily_Return"]).cumprod()

final_strategy_return = df["Strategy_Cum"].iloc[-1] - 1
final_buyhold_return = df["BuyHold_Cum"].iloc[-1] - 1

print("策略最終累積報酬: ", f"{final_strategy_return:.2%}")
print("買入持有累積報酬: ", f"{final_buyhold_return:.2%}")

# 8. 繪圖

# (A) 建立 2x1 的圖表: 上面顯示「價格 + 布林通道 + 買賣點」，下面顯示「MACD 圖」
fig = plt.figure(figsize=(14, 8))

# A1. 上半部：價格 + 布林通道 + 買賣訊號
ax1 = fig.add_subplot(2, 1, 1)
ax1.set_title("USD/CNY (±2 STD)")

ax1.plot(df.index, df["Close"], label="Close", color="blue")
ax1.plot(df.index, df["UpperBand"], label="UpperBand", color="red", linestyle="--", alpha=0.5)
ax1.plot(df.index, df["MA20"], label="MA20", color="green", linestyle="--", alpha=0.5)
ax1.plot(df.index, df["LowerBand"], label="LowerBand", color="red", linestyle="--", alpha=0.5)

buy_signals = df[df["Signal"] == 1]
sell_signals = df[df["Signal"] == -1]
ax1.scatter(buy_signals.index, buy_signals["Close"], color="green", marker="^", s=80, label="Buy")
ax1.scatter(sell_signals.index, sell_signals["Close"], color="red", marker="v", s=80, label="Sell")

ax1.legend(loc="best")
ax1.grid(True)

# A2. 下半部：MACD 圖
ax2 = fig.add_subplot(2, 1, 2)
ax2.set_title("MACD & Signal")
ax2.plot(df.index, df["MACD"], label="MACD", color="blue")
ax2.plot(df.index, df["MACD_Signal"], label="MACD_Signal", color="orange")

# 填充 MACD - Signal 區域 => 方便看出差距正負
ax2.fill_between(df.index, df["MACD"] - df["MACD_Signal"], 0, color="gray", alpha=0.3, label="MACD - Signal")

ax2.legend(loc="best")
ax2.grid(True)
plt.tight_layout()
plt.show()

# (B) 策略累積報酬 vs. BuyHold
plt.figure(figsize=(12, 6))
plt.title("Strategy vs. Buy&Hold")
plt.plot(df.index, df["Strategy_Cum"], label="Strategy", color="blue")
plt.plot(df.index, df["BuyHold_Cum"], label="Buy & Hold", color="gray", linestyle="--")
plt.legend(loc="best")
plt.grid(True)
plt.show()
