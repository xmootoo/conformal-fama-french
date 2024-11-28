import numpy as np
import pandas as pd
import os
from datetime import datetime
from sklearn.linear_model import LinearRegression
from typing import Tuple

def convert_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes date formatting in a DataFrame by renaming and converting the first column.

    Args:
        df (pd.DataFrame): Input DataFrame where first column contains dates
            in YYYYMMDD format (e.g., 20240327)

    Returns:
        pd.DataFrame: DataFrame with first column renamed to 'Date' and converted
            to pandas datetime format
    """
    # Rename the first column to 'Date'
    df.rename(columns={df.columns[0]: 'Date'}, inplace=True)

    # Convert the 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')

    return df

def load_data(
    factors:str="5_factor",
    symbol:str="AAPL",
    pred_len:int=0
) -> pd.DataFrame:
    """
        Loads and merges Fama-French factor data with stock price data.

        Args:
            factors (str, optional): Name of the Fama-French factor model to use.
                Expected format: "{n}_factor". Defaults to "5_factor".
            symbol (str, optional): Stock ticker symbol. Defaults to "AAPL".
            pred_len (int, optional): Prediction length, currently unused. Defaults to 0.

        Returns:
            pd.DataFrame: Merged dataframe containing:
                - Stock data columns: Symbol, Date, Open, High, Low, Close,
                    Adj Close, Volume, Daily Returns
                - Factor data columns: Varies based on chosen factor model
                All data is aligned between the start date of the stock data and
                the earlier of either the stock's end date or the factor data's end date.

        Notes:
            - Requires factor data files to be present in "data/fama_french/{factors}.csv"
            - Requires S&P 500 stock data in "data/sp500/sp500_stocks.csv"
            - Daily returns are calculated using Adjusted Close prices
            - NaN values from return calculations are dropped
            - Dates are converted to datetime format
        """

    # Import Factor Data
    factor_path = os.path.join("data/fama_french", f"{factors}.csv")
    factor_data = convert_date(pd.read_csv(factor_path, skiprows=3))

    print(factor_data.head())

    # Import S&P 500 data
    sp = pd.read_csv("data/sp500/sp500_stocks.csv")

    # Select AAPL stock
    stock_data = sp[sp['Symbol'] == symbol].copy()

    # Calculate daily returns using Adj Close
    stock_data['Daily Returns'] = stock_data['Adj Close'].pct_change()
    stock_data.dropna(inplace=True) # Get rid of first row (NaN)
    stock_data['Date'] = pd.to_datetime(stock_data['Date']) # Ensure date is in datetime format

    # Align factor data with stock data
    start_date = stock_data['Date'].iloc[0]
    end_date = min(
        stock_data['Date'].iloc[-1],
        factor_data['Date'].iloc[-1]
    )

    # Filter dataframes to only include dates between start_date and end_date
    stock_data = stock_data[(stock_data['Date'] >= start_date) & (stock_data['Date'] <= end_date)]
    factor_data = factor_data[(factor_data['Date'] >= start_date) & (factor_data['Date'] <= end_date)]

    # Join the two dataframes on 'Date'
    full_data = pd.merge(stock_data, factor_data, on='Date', how='inner')

    return full_data

def load_forecasting_data(symbol: str, factors: str, seq_len: int, pred_len: int) -> np.ndarray:
    """
    Loads and preprocesses data for price forecasting model.

    Args:
        symbol (str): Stock ticker symbol.
        factors (str): Name of the Fama-French factor model to use.
            Expected format: "{n}_factor".
        seq_len (int): Lookback window size.
        pred_len (int): Forecast horizon size.

    Returns:
        X (np.ndarray): Features array with shape (num_windows, seq_len, n_features).
        y (np.ndarray): Target array with shape (num_windows, pred_len) containing daily returns%.
    """

    # Load and merge data
    df = load_data(factors, symbol, pred_len)
    feature_names = [
        'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF',  # Fama-French factors
        'Daily Returns', 'Volume',
        # Could add derived features like:
        # 'Close/Open', rolling volatility, etc.
    ]
    features = df[feature_names].values
    daily_returns = df['Daily Returns'].values


    X, y = [], []
    for i in range(len(features) - seq_len - pred_len + 1):
        # Input window
        X.append(features[i:i+seq_len])
        # Target: returns for next pred_len days
        y.append(daily_returns[i+seq_len:i+seq_len+pred_len])

    return np.array(X), np.array(y)


def temporal_train_test_split(X, y, train_ratio=0.7, val_ratio=0.15, shuffle=False, seed=42):
   """
   Split data temporally for time series into train/val/test sets with optional shuffling.

   Args:
       X: Shape (n_samples, seq_len, n_features)
       y: Shape (n_samples, pred_len)
       train_ratio: Proportion of data to use for training
       val_ratio: Proportion of data to use for validation
       shuffle: Whether to shuffle the train/val/test sets individually
       seed: Random seed for shuffling
       Note: test_ratio will be 1 - train_ratio - val_ratio
   """
   n_samples = len(X)
   train_size = int(n_samples * train_ratio)
   val_size = int(n_samples * val_ratio)

   # Split into train/val/test in temporal order
   X_train = X[:train_size]
   X_val = X[train_size:train_size+val_size]
   X_test = X[train_size+val_size:]

   y_train = y[:train_size]
   y_val = y[train_size:train_size+val_size]
   y_test = y[train_size+val_size:]

   if shuffle:
       # Shuffle each set independently
       rng = np.random.default_rng(seed)

       # Train
       train_idx = rng.permutation(len(X_train))
       X_train = X_train[train_idx]
       y_train = y_train[train_idx]

       # Validation
       val_idx = rng.permutation(len(X_val))
       X_val = X_val[val_idx]
       y_val = y_val[val_idx]

       # Test
       test_idx = rng.permutation(len(X_test))
       X_test = X_test[test_idx]
       y_test = y_test[test_idx]

   return X_train,y_train, X_val, y_val, X_test, y_test

if __name__ == "__main__":
    factors = "5_factor"
    symbol = "AAPL"
    seq_len = 512
    pred_len = 7
    print(load_data(factors, symbol).head())
    X, y = load_forecasting_data(symbol, factors, seq_len, pred_len)
    print(f"Input shape: {X.shape}. Target shape: {y.shape}")
