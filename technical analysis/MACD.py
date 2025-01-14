import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from typing import Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class TradingParameters:
    initial_investment: float = 1000000
    macd_short_window: int = 12
    macd_long_window: int = 26
    macd_signal_window: int = 9


@dataclass
class StrategyState:
    position: int = 0  # 0: neutral, 1: long
    cash: float = 0
    shares: float = 0
    counter_long_down: int = 0
    counter_long_up: int = 0


class MACDStrategy:
    def __init__(self, params: TradingParameters):
        self.params = params
        self.state = StrategyState(cash=params.initial_investment)
        self.trades: List[Dict] = []

    def calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD and related indicators."""
        df = df.copy()
        df['EMA_short'] = df['Close'].ewm(span=self.params.macd_short_window, adjust=False).mean()
        df['EMA_long'] = df['Close'].ewm(span=self.params.macd_long_window, adjust=False).mean()
        df['MACD'] = df['EMA_short'] - df['EMA_long']
        df['Signal'] = df['MACD'].ewm(span=self.params.macd_signal_window, adjust=False).mean()
        df['Hist'] = df['MACD'] - df['Signal']
        return df

    def execute_buy(self, price: float, date: pd.Timestamp, amount: float = None) -> None:
        """Execute a buy order."""
        price = float(price)  # Convert price to float
        if amount is None:
            amount = self.state.cash
        shares_to_buy = amount // price
        cost = shares_to_buy * price

        if cost <= self.state.cash and cost > 0:
            self.state.cash -= cost
            self.state.shares += shares_to_buy
            self.trades.append({
                'date': date,
                'type': 'buy',
                'price': price,
                'shares': shares_to_buy,
                'value': cost
            })

    def execute_sell(self, price: float, date: pd.Timestamp, percentage: float = 1.0) -> None:
        """Execute a sell order."""
        price = float(price)  # Convert price to float
        shares_to_sell = self.state.shares * percentage
        value = shares_to_sell * price

        if value > 0:
            self.state.cash += value
            self.state.shares -= shares_to_sell
            self.trades.append({
                'date': date,
                'type': 'sell',
                'price': price,
                'shares': shares_to_sell,
                'value': value
            })

    def calculate_portfolio_value(self, price: float) -> float:
        """Calculate current portfolio value."""
        price = float(price)  # Convert price to float
        return self.state.cash + (self.state.shares * price)

    def run_strategy(self, data: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """Execute the MACD trading strategy."""
        strategy_returns = []
        buy_and_hold_returns = []
        initial_price = float(data['Close'].iloc[0])

        for i in range(1, len(data)):
            current_price = float(data['Close'].iloc[i])

            # Calculate buy and hold returns
            buy_and_hold_return = (current_price - initial_price) / initial_price
            buy_and_hold_returns.append(buy_and_hold_return)

            # Get MACD indicators
            prev_hist = float(data['Hist'].iloc[i - 1])
            current_hist = float(data['Hist'].iloc[i])
            macd_value = float(data['MACD'].iloc[i])
            current_date = data.index[i]

            # Trading logic
            if self.state.position == 0:  # Neutral position
                if macd_value < 0 and self.state.counter_long_down == 0 and prev_hist < 0 and current_hist > 0:
                    self.state.counter_long_down += 1
                elif macd_value < 0 and self.state.counter_long_down == 1 and prev_hist < 0 and current_hist > 0:
                    self.state.counter_long_down = 0
                    self.state.position = 1
                    self.execute_buy(current_price, current_date)

            elif self.state.position == 1:  # Long position
                if macd_value < 0 and prev_hist > 0 and current_hist < 0:
                    self.execute_sell(current_price, current_date)
                    self.state.position = 0
                elif macd_value > 0 and self.state.counter_long_up == 0 and prev_hist < 0 and current_hist > 0:
                    self.state.counter_long_up += 1
                    self.execute_buy(current_price, current_date, self.state.cash)
                elif macd_value > 0 and self.state.counter_long_up == 1 and prev_hist < 0 and current_hist > 0:
                    self.state.counter_long_up += 1
                    self.execute_sell(current_price, current_date, 0.5)
                elif macd_value > 0 and prev_hist > 0 and current_hist < 0:
                    self.execute_sell(current_price, current_date)
                    self.state.position = 0

            # Calculate strategy returns
            portfolio_value = self.calculate_portfolio_value(current_price)
            strategy_return = (portfolio_value - self.params.initial_investment) / self.params.initial_investment
            strategy_returns.append(strategy_return)

        return strategy_returns, buy_and_hold_returns

    def calculate_metrics(self, strategy_returns: List[float], buy_and_hold_returns: List[float]) -> Dict:
        """Calculate trading strategy metrics."""
        strategy_returns_arr = np.array(strategy_returns)

        # Calculate basic metrics
        final_strategy_return = strategy_returns[-1]
        final_buy_and_hold_return = buy_and_hold_returns[-1]

        # Calculate maximum drawdown
        cumulative_returns = np.maximum.accumulate(strategy_returns_arr)
        drawdowns = (cumulative_returns - strategy_returns_arr) / np.maximum(cumulative_returns, 1e-10)
        max_drawdown = np.max(drawdowns)

        # Calculate win rate
        daily_returns = np.diff(strategy_returns_arr)
        win_trades = np.sum(daily_returns > 0)
        win_rate = win_trades / len(daily_returns)

        # Calculate Sharpe Ratio (assuming 252 trading days per year)
        returns_std = np.std(daily_returns)
        sharpe_ratio = np.mean(daily_returns) / returns_std * np.sqrt(252) if returns_std > 0 else 0

        return {
            'strategy_return': final_strategy_return,
            'buy_and_hold_return': final_buy_and_hold_return,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': len(self.trades)
        }


def plot_results(strategy_returns: List[float], buy_and_hold_returns: List[float],
                 metrics: Dict, title: str = 'TWII MACD Trading Strategy Results'):
    """Plot strategy results with metrics annotation."""
    plt.figure(figsize=(15, 8))
    plt.plot(strategy_returns, label='Strategy Returns', alpha=0.7)
    plt.plot(buy_and_hold_returns, label='Buy & Hold Returns', alpha=0.7)

    # Add metrics annotation
    metrics_text = (f"Strategy Return: {metrics['strategy_return']:.2%}\n"
                    f"Buy & Hold Return: {metrics['buy_and_hold_return']:.2%}\n"
                    f"Max Drawdown: {metrics['max_drawdown']:.2%}\n"
                    f"Win Rate: {metrics['win_rate']:.2%}\n"
                    f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
                    f"Total Trades: {metrics['total_trades']}")

    plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Returns')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()


def main():
    # Download data
    symbol = "TSLA"
    start_date = "2020-01-01"
    end_date = "2024-01-01"
    data = yf.download(symbol, start=start_date, end=end_date, interval="1d")

    if data.empty:
        raise ValueError("无法获取数据，请检查网络或时间区间。")

    # Initialize and run strategy
    params = TradingParameters()
    strategy = MACDStrategy(params)
    data = strategy.calculate_macd(data)
    strategy_returns, buy_and_hold_returns = strategy.run_strategy(data)

    # Calculate and display metrics
    metrics = strategy.calculate_metrics(strategy_returns, buy_and_hold_returns)
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            print(
                f"{metric_name}: {value:.2%}" if 'return' in metric_name or 'rate' in metric_name else f"{metric_name}: {value:.2f}")
        else:
            print(f"{metric_name}: {value}")

    # Plot results
    plot_results(strategy_returns, buy_and_hold_returns, metrics)
    plt.show()


if __name__ == "__main__":
    main()