import pandas as pd
import numpy as np

def calculate_rsi(df, column='close', period=14):
    delta = df[column].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(df, column='close', short_period=12, long_period=26, signal_period=9):
    short_ema = df[column].ewm(span=short_period, adjust=False).mean()
    long_ema = df[column].ewm(span=long_period, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

def calculate_vamp(df, price_column='close', volume_column='volume', period=14):
    vamp = (df[price_column] * df[volume_column]).rolling(window=period).sum() / df[volume_column].rolling(window=period).sum()
    return vamp

# Example usage with a DataFrame `df`
# Creating a sample DataFrame for demonstration
data = {
    'Gmt time': pd.date_range(start='2022-01-01', periods=30, freq='D'),
    'open': np.random.rand(30) * 100,
    'close': np.random.rand(30) * 100,
    'high': np.random.rand(30) * 100,
    'low': np.random.rand(30) * 100,
    'volume': np.random.randint(1, 1000, 30)
}

df = pd.DataFrame(data)

# Calculate RSI and add to DataFrame
df['RSI'] = calculate_rsi(df)

# Calculate MACD and add to DataFrame
df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df)

# Calculate VAMP and add to DataFrame
df['VAMP'] = calculate_vamp(df)

print(df)