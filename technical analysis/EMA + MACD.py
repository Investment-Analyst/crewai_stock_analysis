import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# 1. 下載/載入 USD/CNY (示例: 離岸人民幣 CNH=X)
# --------------------------
symbol = "CNY=X"
start_date = "2020-01-01"
end_date = "2024-12-28"

df = yf.download(symbol, start=start_date, end=end_date)
df.dropna(inplace=True)
df['Price'] = df['Close']  # 收盤價

# --------------------------
# 2. 計算技術指標 (MA, MACD, RSI, Bollinger Bands)
# --------------------------
short_window = 20
long_window = 50

# 移動平均
df['MA_short'] = df['Price'].rolling(window=short_window).mean()
df['MA_long'] = df['Price'].rolling(window=long_window).mean()

# MACD
exp1 = df['Price'].ewm(span=13, adjust=False).mean()
exp2 = df['Price'].ewm(span=26, adjust=False).mean()
df['MACD'] = exp1 - exp2
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

# RSI
window_rsi = 14
delta = df['Price'].diff()
gain = np.where(delta > 0, delta, 0)
loss = np.where(delta < 0, -delta, 0)
avg_gain = pd.Series(gain).rolling(window=window_rsi).mean()
avg_loss = pd.Series(loss).rolling(window=window_rsi).mean()
rs = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))

# 布林通道
df['std'] = df['Price'].rolling(window=short_window).std()
df['Boll_Upper'] = df['MA_short'] + 2 * df['std']
df['Boll_Lower'] = df['MA_short'] - 2 * df['std']

# --------------------------
# 3. 建立交易信號 (MA, MACD)
# --------------------------
# Signal generation mechanism:
# - A "Buy" signal is generated when the short-term moving average (MA_short) crosses above the long-term moving average (MA_long),
#   and the MACD value is greater than its signal line (MACD_Signal). This indicates a potential upward trend.
# - A "Sell" signal is generated when the short-term moving average (MA_short) crosses below the long-term moving average (MA_long),
#   and the MACD value is less than its signal line (MACD_Signal). This indicates a potential downward trend.
df['Signal'] = 0
df.loc[
    (df['MA_short'] > df['MA_long']) & (df['MACD'] > df['MACD_Signal']),
    'Signal'
] = 1  # Buy 信號

df.loc[
    (df['MA_short'] < df['MA_long']) & (df['MACD'] < df['MACD_Signal']),
    'Signal'
] = -1  # Sell 信號

df['Crossover'] = df['Signal'].diff()

# --------------------------
# 4. 回測邏輯
# --------------------------
position = 0
entry_price = None
contract_size = 300000
initial_capital = 1000000

trade_records = []
daily_values = [initial_capital]

for i in range(1, len(df)):
    date_i = df.index[i]
    price_i = df['Price'].iloc[i]
    crossover = df['Crossover'].iloc[i]
    signal_i = df['Signal'].iloc[i]

    # 添加倉位檢查以防止同方向信號的重複執行
    if crossover != 0:
        if signal_i == 1 and position != 1:  # Buy 信號，只在非多頭時執行
            if position == -1:  # 平空
                pnl = (entry_price - price_i) * contract_size
                trade_records.append({'Date': date_i, 'Action': 'Close Short', 'Price': price_i, 'PnL': pnl})
            position = 1
            entry_price = price_i
            trade_records.append({'Date': date_i, 'Action': 'Buy', 'Price': price_i, 'PnL': 0})
        elif signal_i == -1 and position != -1:  # Sell 信號，只在非空頭時執行
            if position == 1:  # 平多
                pnl = (price_i - entry_price) * contract_size
                trade_records.append({'Date': date_i, 'Action': 'Close Long', 'Price': price_i, 'PnL': pnl})
            position = -1
            entry_price = price_i
            trade_records.append({'Date': date_i, 'Action': 'Sell', 'Price': price_i, 'PnL': 0})

    daily_value = initial_capital + sum([rec['PnL'] for rec in trade_records])
    if position == 1:
        daily_value += (price_i - entry_price) * contract_size
    elif position == -1:
        daily_value += (entry_price - price_i) * contract_size
    daily_values.append(daily_value)

if position != 0:
    final_price = df['Price'].iloc[-1]
    final_date = df.index[-1]
    if position == 1:
        pnl = (final_price - entry_price) * contract_size
        trade_records.append({'Date': final_date, 'Action': 'Close Long (Final)', 'Price': final_price, 'PnL': pnl})
    elif position == -1:
        pnl = (entry_price - final_price) * contract_size
        trade_records.append({'Date': final_date, 'Action': 'Close Short (Final)', 'Price': final_price, 'PnL': pnl})

# --------------------------
# 5. 計算策略績效指標
# --------------------------
total_pnl = sum([rec['PnL'] for rec in trade_records])
final_strategy_value = initial_capital + total_pnl
strategy_return_percent = (final_strategy_value - initial_capital) / initial_capital * 100

# 最大回撤
daily_values = np.array(daily_values)
drawdowns = (daily_values / np.maximum.accumulate(daily_values)) - 1
max_drawdown = drawdowns.min()

# 勝率
winning_trades = [rec for rec in trade_records if rec['PnL'] > 0]
win_rate = len(winning_trades) / len(trade_records) * 100 if trade_records else 0

# 風險報酬比
annualized_return = (1 + strategy_return_percent / 100) ** (1 / (len(df) / 252)) - 1
annualized_volatility = np.std(drawdowns) * np.sqrt(252)
risk_reward_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else np.nan

# --------------------------
# 6. 繪圖
# --------------------------
fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

# 主圖
ax1.plot(df.index, df['Price'], label='Price', color='black')
ax1.plot(df.index, df['MA_short'], label=f'MA_{short_window}', color='blue')
ax1.plot(df.index, df['MA_long'], label=f'MA_{long_window}', color='orange')
ax1.fill_between(df.index, df['Boll_Lower'], df['Boll_Upper'], color='green', alpha=0.1, label='Bollinger Bands')
for r in trade_records:
    if 'Buy' in r['Action']:
        ax1.scatter(r['Date'], r['Price'], marker='^', color='red', s=100)
    elif 'Sell' in r['Action']:
        ax1.scatter(r['Date'], r['Price'], marker='v', color='green', s=100)
ax1.legend()
ax1.set_title('Price, MA, and Bollinger Bands')

# MACD 圖
ax3.plot(df.index, df['MACD'], label='MACD', color='blue')
ax3.plot(df.index, df['MACD_Signal'], label='MACD Signal', color='red')
ax3.bar(df.index, df['MACD_Hist'], color='gray', alpha=0.5, label='MACD Histogram')
ax3.legend()
ax3.set_title('MACD')

plt.tight_layout()
plt.show()

# --------------------------
# 7. 列印回測結果
# --------------------------
print("=== 交易紀錄 ===")
for r in trade_records:
    print(f"{r['Date']} | {r['Action']:<15} | Price={r['Price']:.4f} | 單次盈虧={r['PnL']:.2f}")

print("\n=== 回測結果 ===")
print(f"最終策略資產價值: {final_strategy_value:,.2f}")
print(f"策略報酬率: {strategy_return_percent:.2f} %")
print(f"最大回撤: {max_drawdown:.2%}")
print(f"勝率: {win_rate:.2f} %")
print(f"風險報酬比: {risk_reward_ratio:.2f}")
print(f"年化報酬率: {annualized_return:.2%}")
print(f"年化波動率: {annualized_volatility:.2%}")
