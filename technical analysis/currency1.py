import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 設定資料期間
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


# 計算EMA (例如 20日、50日)
data['EMA20'] = data['Close'].ewm(span=20, adjust=False).mean()
data['EMA50'] = data['Close'].ewm(span=50, adjust=False).mean()

# 計算指標
data['RSI'] = RSI(data['Close'], 14)
data['MACD_line'], data['Signal_line'], data['Hist'] = MACD(data['Close'])

# ---------------------
# 建立策略邏輯(簡化版)：同前回測程式，將交易時機記錄下來用於繪圖
# ---------------------
position = 0
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

        # 若已在多單，遇死亡交叉 + RSI>70 平多並反向做空 (簡化策略)
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

        # 若已在空單，遇黃金交叉 + RSI<30 平空並反向做多 (簡化策略)
        if position == -1 and golden_cross and rsi < 30:
            exit_price = close_price
            pnl = entry_price - exit_price
            trades.append({'Date': date, 'Type': 'Buy_to_close', 'Price': exit_price, 'PnL': pnl})
            position = 0
            # 做多
            position = 1
            entry_price = close_price
            trades.append({'Date': date, 'Type': 'Buy', 'Price': entry_price})
            rsi_oversold = False

# 最後持倉平倉
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

# ---------------------
# 繪圖
# ---------------------
fig = plt.figure(figsize=(14, 10))
ax1 = plt.subplot2grid((3, 1), (0, 0))
ax2 = plt.subplot2grid((3, 1), (1, 0))
ax3 = plt.subplot2grid((3, 1), (2, 0))

# 第一張子圖：價格、EMA及買賣點
ax1.plot(data.index, data['Close'], label='Close Price')
ax1.plot(data.index, data['EMA20'], label='EMA20', alpha=0.7)
ax1.plot(data.index, data['EMA50'], label='EMA50', alpha=0.7)

# 在價格圖上標記交易點
buy_signals = trade_df[(trade_df['Type'] == 'Buy') | (trade_df['Type'] == 'Buy_to_close')]
sell_signals = trade_df[(trade_df['Type'] == 'Sell') | (trade_df['Type'] == 'Sell_to_close')]

ax1.scatter(buy_signals['Date'], buy_signals['Price'], marker='^', color='green', s=100, label='Buy signals')
ax1.scatter(sell_signals['Date'], sell_signals['Price'], marker='v', color='red', s=100, label='Sell signals')

ax1.set_title('USD/CNY Price with Trades')
ax1.set_ylabel('Price (USD/CNY)')
ax1.legend()
ax1.grid(True)

# 第二張子圖：RSI
ax2.plot(data.index, data['RSI'], label='RSI', color='orange')
ax2.axhline(70, color='red', linestyle='--', alpha=0.5)
ax2.axhline(30, color='green', linestyle='--', alpha=0.5)
ax2.set_title('RSI(14)')
ax2.set_ylabel('RSI')
ax2.grid(True)
ax2.legend()

# 第三張子圖：MACD
ax3.plot(data.index, data['MACD_line'], label='MACD_line', color='blue')
ax3.plot(data.index, data['Signal_line'], label='Signal_line', color='red')
ax3.bar(data.index, data['Hist'], label='MACD Hist', color='gray', alpha=0.5)
ax3.set_title('MACD(12,26,9)')
ax3.set_ylabel('Value')
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.show()

