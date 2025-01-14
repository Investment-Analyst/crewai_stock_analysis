import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# 設定資料期間(從2020-01-01至今)
start_date = "2020-01-01"
end_date = datetime.today().strftime("%Y-%m-%d")

# 下載USD/CNY日線匯率資料 (CNY=X 是USD/CNY)
data = yf.download("CNY=X", start=start_date, end=end_date)

if data.empty:
    raise ValueError("無法取得價格資料，請檢查代碼或日期範圍")


# 計算RSI (14日)
def RSI(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# 計算MACD (12,26,9)
def MACD(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


data['RSI'] = RSI(data['Close'], 14)
data['MACD_line'], data['Signal_line'], data['Hist'] = MACD(data['Close'])

# 回測策略
position = 0  # 0: 無部位, 1: 做多, -1: 做空
entry_price = 0.0
trades = []
rsi_overbought = False
rsi_oversold = False

for i in range(len(data)):
    rsi = data['RSI'].iloc[i]
    macd = data['MACD_line'].iloc[i]
    signal = data['Signal_line'].iloc[i]
    close_price = data['Close'].iloc[i]
    date = data.index[i]

    if rsi > 50:
        rsi_overbought = True
    elif rsi < 50:
        rsi_oversold = True

    if i > 0:
        macd_yesterday = data['MACD_line'].iloc[i - 1]
        signal_yesterday = data['Signal_line'].iloc[i - 1]

        golden_cross = (macd_yesterday < signal_yesterday) and (macd > signal)
        death_cross = (macd_yesterday > signal_yesterday) and (macd < signal)

        # RSI < 30後遇黃金交叉，買進
        if golden_cross and rsi_oversold and position == 0:
            position = 1
            entry_price = close_price
            rsi_oversold = False
            trades.append({'Date': date, 'Type': 'Buy', 'Price': entry_price})

        # RSI > 70後遇死亡交叉，賣出
        if death_cross and rsi_overbought and position == 0:
            position = -1
            entry_price = close_price
            rsi_overbought = False
            trades.append({'Date': date, 'Type': 'Sell', 'Price': entry_price})

        # 多單轉空單
        if position == 1 and death_cross and rsi > 70:
            exit_price = close_price
            pnl = exit_price - entry_price
            trades.append({'Date': date, 'Type': 'Sell_to_close', 'Price': exit_price, 'PnL': pnl})
            position = 0
            # 進空
            position = -1
            entry_price = close_price
            trades.append({'Date': date, 'Type': 'Sell', 'Price': entry_price})
            rsi_overbought = False

        # 空單轉多單
        if position == -1 and golden_cross and rsi < 30:
            exit_price = close_price
            pnl = entry_price - exit_price
            trades.append({'Date': date, 'Type': 'Buy_to_close', 'Price': exit_price, 'PnL': pnl})
            position = 0
            # 進多
            position = 1
            entry_price = close_price
            trades.append({'Date': date, 'Type': 'Buy', 'Price': entry_price})
            rsi_oversold = False

# 最後持倉在回測結束日平倉
if position != 0:
    exit_price = data['Close'].iloc[-1]
    if position == 1:
        pnl = exit_price - entry_price
        trades.append({'Date': data.index[-1], 'Type': 'Final_close_buy', 'Price': exit_price, 'PnL': pnl})
    else:
        pnl = entry_price - exit_price
        trades.append({'Date': data.index[-1], 'Type': 'Final_close_sell', 'Price': exit_price, 'PnL': pnl})
    position = 0

trade_df = pd.DataFrame(trades)

print("交易紀錄：")
print(trade_df)

# 計算策略總PnL
if 'PnL' in trade_df.columns:
    strategy_pnl = trade_df['PnL'].sum(skipna=True)
else:
    strategy_pnl = 0.0

print(f"策略總損益 (CNY): {strategy_pnl}")

# 計算基準：買入並持有策略
# 假設在2021-01-01以當日收盤價買入1 USD，最後以最後收盤價賣出
start_price = data['Close'].iloc[0]
end_price = data['Close'].iloc[-1]
benchmark_pnl = end_price - start_price
print(f"基準買入並持有策略損益 (CNY)：{benchmark_pnl}")

strategy_pnl = float(trade_df['PnL'].sum(skipna=True))
benchmark_pnl = float(end_price - start_price)


# 比較策略與基準
# 策略pnl > 基準pnl 則策略表現優於大盤，反之劣於
if strategy_pnl > benchmark_pnl:
    print("策略表現優於大盤(買入並持有)。")
elif strategy_pnl < benchmark_pnl:
    print("策略表現不如大盤(買入並持有)。")
else:
    print("策略表現與大盤相當。")

# 繪製最後的價格與交易點可視化 (簡單示意)
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label='Close Price')
buy_points = trade_df[trade_df['Type'].str.contains('Buy')].dropna()
sell_points = trade_df[trade_df['Type'].str.contains('Sell')].dropna()

plt.scatter(buy_points['Date'], buy_points['Price'], marker='^', color='green', s=100, label='Buy Points')
plt.scatter(sell_points['Date'], sell_points['Price'], marker='v', color='red', s=100, label='Sell Points')

plt.title('USD/CNY Close Price with Trading Signals')
plt.xlabel('Date')
plt.ylabel('Price (CNY per USD)')
plt.legend()
plt.grid(True)
plt.show()
