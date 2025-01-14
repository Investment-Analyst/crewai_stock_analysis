import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
import yfinance as yf
import traceback
import sys


def download_data(ticker, years=3):
    """
    下載指定年數的比特幣歷史數據

    Parameters:
        ticker (str): 交易對符號
        years (int): 要下載的年數

    Returns:
        pandas.DataFrame: 包含OHLCV數據的DataFrame
    """
    import datetime
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=years * 365)

    print(f"下載 {ticker} 數據從 {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}")

    data = yf.download(
        tickers=ticker,
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d'),
        interval='1d'
    )

    if data.empty:
        raise ValueError("無法下載數據。請檢查交易對符號和網絡連接。")

    data.dropna(inplace=True)
    return data


def calculate_indicators(data):
    """
    Calculate technical indicators with improved data alignment and error handling

    Parameters:
        data (pandas.DataFrame): Raw OHLCV data

    Returns:
        pandas.DataFrame: Data with calculated technical indicators
    """
    # Create a copy to avoid modifying original data
    result = data.copy()

    # Ensure all input data are float type
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        result[col] = result[col].astype(float)

    # Calculate moving averages and standard deviation
    result['MA20'] = result['Close'].rolling(window=20).mean()
    result['STD'] = result['Close'].rolling(window=20).std()

    # Calculate Bollinger Bands
    result['Upper_Band'] = result['MA20'] + (result['STD'] * 2.5)
    result['Lower_Band'] = result['MA20'] - (result['STD'] * 2.5)

    # Calculate %B - Key Fix: Ensure we're working with Series objects
    try:
        # Extract Close price as Series
        close_series = result['Close'].squeeze()
        lower_series = result['Lower_Band'].squeeze()
        upper_series = result['Upper_Band'].squeeze()

        # Calculate %B using Series operations
        result['%B'] = (close_series - lower_series) / (upper_series - lower_series)

        # Verify calculation
        print(f"%B calculation shape: {result['%B'].shape}")
        print(f"First few %B values: {result['%B'].head()}")

    except Exception as e:
        print("\nDiagnostic information:")
        print(f"Close shape: {result['Close'].shape}")
        print(f"Lower_Band shape: {result['Lower_Band'].shape}")
        print(f"Upper_Band shape: {result['Upper_Band'].shape}")
        raise Exception(f"Error calculating %B: {str(e)}")

    # Calculate RSI
    delta = result['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    result['RSI'] = 100 - (100 / (1 + rs))

    # Calculate MACD
    result['MACD'] = (result['Close'].ewm(span=12, adjust=False).mean() -
                      result['Close'].ewm(span=26, adjust=False).mean())
    result['Signal_Line'] = result['MACD'].ewm(span=9, adjust=False).mean()

    # Calculate Momentum
    result['Momentum'] = result['Close'].diff(periods=10)

    # Calculate ATR
    tr1 = result['High'] - result['Low']
    tr2 = abs(result['High'] - result['Close'].shift())
    tr3 = abs(result['Low'] - result['Close'].shift())
    result['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    result['ATR'] = result['TR'].rolling(window=14).mean()

    # Clean data
    result = result.replace([np.inf, -np.inf], np.nan)
    initial_rows = len(result)
    result = result.dropna()
    final_rows = len(result)

    print(f"\nData cleaning report:")
    print(f"Initial rows: {initial_rows}")
    print(f"Final rows: {final_rows}")
    print(f"Rows removed: {initial_rows - final_rows}")

    return result


def create_sequences(data, seq_length):
    """
    創建用於LSTM的序列數據

    Parameters:
        data (numpy.array): 標準化後的特徵數據
        seq_length (int): 序列長度

    Returns:
        tuple: (X序列數據, y標籤數據)
    """
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i])
        y.append(data[i, 0])  # 預測收盤價
    return np.array(X), np.array(y)


def build_train_lstm(X_train, y_train, X_test, y_test):
    """
    構建並訓練LSTM模型

    Parameters:
        X_train, y_train: 訓練數據
        X_test, y_test: 測試數據

    Returns:
        keras.Model: 訓練好的模型
    """
    model = Sequential([
        LSTM(128, return_sequences=True,
             input_shape=(X_train.shape[1], X_train.shape[2]),
             kernel_regularizer=l2(0.001)),
        Dropout(0.3),
        LSTM(64, return_sequences=False,
             kernel_regularizer=l2(0.001)),
        Dropout(0.3),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    print("開始訓練LSTM模型...")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )

    return model


def implement_strategy(data, predictions, ticker, initial_capital=500):
    """
    实现交易策略，支持做空逻辑

    Parameters:
        data (pandas.DataFrame): 市场数据，可能带有多层索引
        predictions (numpy.array): 模型预测值
        ticker (str): 股票代码
        initial_capital (float): 初始资金

    Returns:
        pandas.DataFrame: 包含交易信号和组合价值的DataFrame
    """
    data = data.copy()

    # 确保 predictions 是一维数组
    if len(predictions.shape) > 1 and predictions.shape[1] == 1:
        predictions = predictions.flatten()
    data['Predicted_Close'] = predictions

    # 选择 'Close' 列
    if isinstance(data.columns, pd.MultiIndex):
        data['Close'] = data[('Close', ticker)]
        data = data.droplevel(level=1, axis=1)

    # 计算预测变化率
    data['Predicted_Change'] = (data['Predicted_Close'] - data['Close']) / data['Close']

    # 动态阈值
    rsi_buy_threshold = data['RSI'].quantile(0.4)
    rsi_sell_threshold = data['RSI'].quantile(0.6)
    pred_change_buy_threshold = data['Predicted_Change'].quantile(0.6)
    pred_change_sell_threshold = data['Predicted_Change'].quantile(0.4)

    # 生成交易信号
    data['Signal'] = 0
    data.loc[(data['Predicted_Change'] > pred_change_buy_threshold) &
             (data['RSI'] < rsi_buy_threshold), 'Signal'] = 1
    data.loc[(data['Predicted_Change'] < pred_change_sell_threshold) &
             (data['RSI'] > rsi_sell_threshold), 'Signal'] = -1

    # 模拟交易
    cash = initial_capital
    holdings = 0
    short_holdings = 0  # 用于记录做空的头寸
    portfolio_value = []
    transaction_cost = 0.0005  # 0.05% 交易成本

    for _, row in data.iterrows():
        price = row['Close']
        signal = row['Signal']

        # 买入信号
        if signal == 1:
            amount_to_buy = (cash * 0.5) * (1 - transaction_cost)
            holdings += amount_to_buy / price
            cash -= amount_to_buy

        # 卖出信号（做空）
        elif signal == -1:
            amount_to_short = (cash * 0.5) * (1 - transaction_cost)
            short_holdings += amount_to_short / price
            cash += amount_to_short

        # 计算组合总价值
        total_value = cash + holdings * price - short_holdings * price
        portfolio_value.append(total_value)

    data['Portfolio_Value'] = portfolio_value
    return data


def evaluate_performance(data, initial_capital=500):
    """
    評估策略性能

    Parameters:
        data (pandas.DataFrame): 包含交易結果的DataFrame
        initial_capital (float): 初始資金
    """
    # 計算回報率
    data['Daily_Return'] = data['Portfolio_Value'].pct_change()
    data['Cumulative_Return'] = (1 + data['Daily_Return']).cumprod()

    # 計算各種績效指標
    total_return = ((data['Portfolio_Value'].iloc[-1] - initial_capital) /
                    initial_capital) * 100
    annualized_return = (data['Cumulative_Return'].iloc[-1] **
                         (365 / len(data)) - 1) * 100

    returns = data['Daily_Return'].dropna()
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)

    # 計算最大回撤
    rolling_max = data['Portfolio_Value'].cummax()
    drawdown = data['Portfolio_Value'] / rolling_max - 1
    max_drawdown = drawdown.min() * 100

    # 計算勝率
    winning_trades = data[data['Signal'] == 1]['Portfolio_Value'].pct_change()
    losing_trades = data[data['Signal'] == -1]['Portfolio_Value'].pct_change()
    win_rate = len(winning_trades[winning_trades > 0]) / max(len(winning_trades), 1)

    # 打印結果
    print("\n==== 策略績效評估 ====")
    print(f"總回報率: {total_return:.2f}%")
    print(f"年化回報率: {annualized_return:.2f}%")
    print(f"夏普比率: {sharpe_ratio:.2f}")
    print(f"最大回撤: {max_drawdown:.2f}%")
    print(f"勝率: {win_rate:.2%}")
    print(f"風險報酬比: {sharpe_ratio:.2f}")


def plot_results(data):
    """
    繪製結果圖表

    Parameters:
        data (pandas.DataFrame): 包含交易結果的DataFrame
    """
    # 确保数据完整性
    if 'Portfolio_Value' not in data.columns:
        print("Error: 'Portfolio_Value' column not found in data.")
        return
    if data['Portfolio_Value'].isna().any():
        print("Warning: 'Portfolio_Value' contains NaN values.")
        return

    # 繪製資產變化圖
    plt.figure(figsize=(15, 7))
    plt.plot(data.index, data['Portfolio_Value'],
             label='Portfolio Value', color='blue')
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 绘制信号和价格
    plt.figure(figsize=(15, 7))
    plt.plot(data.index, data['Close'], label='Close Price', color='blue')
    buy_points = data[data['Signal'] == 1].index
    sell_points = data[data['Signal'] == -1].index

    plt.scatter(buy_points, data.loc[buy_points, 'Close'],
                color='green', marker='^', s=100, label='Buy Signal')
    plt.scatter(sell_points, data.loc[sell_points, 'Close'],
                color='red', marker='v', s=100, label='Sell Signal')

    plt.title('Stock Price with Trading Signals')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()



# 主程序
def main():
    try:
        print("開始執行交易策略...")

        # 定義交易股票代號
        ticker = 'NVDA'

        # 下載數據
        data = download_data(ticker=ticker, years=5)
        print("數據下載完成，開始計算技術指標...")

        # 計算技術指標
        data = calculate_indicators(data)

        # 準備特徵數據
        features = ['Close', '%B', 'RSI', 'MACD', 'Signal_Line', 'Momentum', 'ATR']
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data[features])

        # 創建序列數據
        seq_length = 60
        X, y = create_sequences(scaled_data, seq_length)
        print(f"創建的序列數據形狀: X={X.shape}, y={y.shape}")

        # 分割數據集
        train_size = int(len(X) * 5 / 7)
        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:], y[train_size:]

        # 訓練模型
        model = build_train_lstm(X_train, y_train, X_test, y_test)

        # 預測
        print("生成預測結果...")
        predictions = model.predict(X_test)

        # 實現交易策略，傳遞 `ticker`
        test_data = data.iloc[train_size + seq_length:]
        test_data = implement_strategy(test_data, predictions, ticker)

        # 評估績效
        evaluate_performance(test_data)

        # 繪製結果
        plot_results(test_data)

        print("策略回測完成!")

    except Exception as e:
        print(f"執行過程中發生錯誤: {str(e)}")
        print("詳細錯誤信息:")
        print(traceback.format_exc())



if __name__ == "__main__":
    main()