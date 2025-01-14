import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------
# 1. 從 Yahoo Finance 下載 CNY=X 日線資料
#------------------------------------------
symbol = "TSLA"
start_date = "2021-01-01"
end_date = "2024-01-01"

data = yf.download(symbol, start=start_date, end=end_date, interval="1d")

if data.empty:
    raise ValueError("無法取得資料，請確認時間區間或網路是否正常。")

#------------------------------------------
# 2. 計算 Momentum & OBV
#------------------------------------------
momentum_period = 10  # 設定 Momentum 週期，可自行調整
data['Momentum'] = data['Close'].diff(momentum_period)

# 計算 OBV
obv = [0]
close_prices = data['Close'].values
volumes = data['Volume'].values

for i in range(1, len(data)):
    if close_prices[i] > close_prices[i - 1]:
        obv.append(obv[-1] + volumes[i])
    elif close_prices[i] < close_prices[i - 1]:
        obv.append(obv[-1] - volumes[i])
    else:
        obv.append(obv[-1])

data['OBV'] = obv

#------------------------------------------
# 3. 產生買/賣訊號
#------------------------------------------
def generate_signals(df):
    buy_signals = []
    sell_signals = []

    momentum_arr = df['Momentum'].values
    obv_arr = df['OBV'].values
    close_arr = df['Close'].values

    for i in range(len(df)):
        if i == 0:
            buy_signals.append(np.nan)
            sell_signals.append(np.nan)
            continue

        # Buy 條件：Momentum > 0 & OBV 上升
        if (momentum_arr[i] > 0) and (obv_arr[i] > obv_arr[i - 1]):
            buy_signals.append(close_arr[i])
            sell_signals.append(np.nan)
        # Sell 條件：Momentum < 0
        elif momentum_arr[i] < 0:
            buy_signals.append(np.nan)
            sell_signals.append(close_arr[i])
        else:
            buy_signals.append(np.nan)
            sell_signals.append(np.nan)

    df['Buy_Signal'] = buy_signals
    df['Sell_Signal'] = sell_signals
    return df

data = generate_signals(data)

#------------------------------------------
# 4. 視覺化買/賣訊號
#------------------------------------------
plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='Close Price', alpha=0.5)
plt.scatter(data.index, data['Buy_Signal'], label='Buy Signal',
            marker='^', color='green', s=50)
plt.scatter(data.index, data['Sell_Signal'], label='Sell Signal',
            marker='v', color='red', s=50)
plt.title(f"Momentum + OBV Trading Signals for {symbol}")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

#------------------------------------------
# 5. 回測：多/空雙向操作 (Long/Short)
#------------------------------------------
data['Position'] = 0  # 初始沒有定義，後面直接根據信號切換

for i in range(1, len(data)):
    prev_pos = data['Position'].iloc[i-1]
    buy_signal = data['Buy_Signal'].iloc[i]
    sell_signal = data['Sell_Signal'].iloc[i]

    if not np.isnan(buy_signal):
        # 收到買進訊號 => 切換為做多 +1
        data.loc[data.index[i], 'Position'] = 1
    elif not np.isnan(sell_signal):
        # 收到賣出訊號 => 直接做空 -1
        data.loc[data.index[i], 'Position'] = -1
    else:
        # 若當天沒信號，沿用前一天的持倉
        data.loc[data.index[i], 'Position'] = prev_pos

#------------------------------------------
# 6. 計算策略績效
#------------------------------------------
data['Daily_Return'] = data['Close'].pct_change()
data['Strategy_Return'] = data['Position'].shift(1) * data['Daily_Return']

# 累計報酬
data['Cumulative_Market'] = (1 + data['Daily_Return']).cumprod()
data['Cumulative_Strategy'] = (1 + data['Strategy_Return']).cumprod()

#------------------------------------------
# 7. 計算最大回撤
#------------------------------------------
cum_returns = data['Cumulative_Strategy']
rolling_max = cum_returns.cummax()
drawdown = (cum_returns - rolling_max) / rolling_max
max_drawdown = drawdown.min()

#------------------------------------------
# 8. 計算勝率
#------------------------------------------
data['Trades'] = data['Position'].diff().fillna(0).abs()
trades = data[data['Trades'] > 0]
positive_trades = trades[trades['Strategy_Return'] > 0]
win_rate = len(positive_trades) / len(trades) if len(trades) > 0 else 0

#------------------------------------------
# 9. 計算風險報酬比 (Sharpe Ratio)
#------------------------------------------
average_return = data['Strategy_Return'].mean()
volatility = data['Strategy_Return'].std()
sharpe_ratio = average_return / volatility * np.sqrt(252)  # 年化Sharpe Ratio

#------------------------------------------
# 10. 繪製累計報酬曲線
#------------------------------------------
plt.figure(figsize=(12,6))
plt.plot(data['Cumulative_Market'], label='Buy & Hold')
plt.plot(data['Cumulative_Strategy'], label='Momentum Strategy')
plt.title("Cumulative Returns Comparison")
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.legend()
plt.show()

#------------------------------------------
# 11. 列印績效數據
#------------------------------------------
final_strategy_return = data['Cumulative_Strategy'].iloc[-1] - 1
final_market_return = data['Cumulative_Market'].iloc[-1] - 1

print(f"交易區間: {start_date} ~ {end_date}")
print(f"{symbol} 動量策略報酬率: {final_strategy_return:.2%}")
print(f"買進持有報酬率: {final_market_return:.2%}")
print(f"最大回撤: {max_drawdown:.2%}")
print(f"勝率: {win_rate:.2%}")
print(f"風險報酬比: {sharpe_ratio:.2f}")
