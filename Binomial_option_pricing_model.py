import pandas as pd
import numpy as np
df = pd.read_parquet('/Users/banti/Downloads/nifty/NIFTY_2024-02-20.parquet')
def calculate_days_to_maturity(expiry_date, p_date):
    return (expiry_date - p_date).days / 365  # Convert to years

# Binomial Option Pricing Model function
def binomial_option_pricing(S, K, T, r, sigma, n, option_type='PE'):
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    # Initialize asset prices and option prices
    asset_prices = np.zeros(n+1)
    option_prices = np.zeros(n+1)

    for i in range(n+1):
        asset_prices[i] = S * (u ** (n-i)) * (d ** i)
        if option_type == 'PE':
            option_prices[i] = max(K - asset_prices[i], 0)

    # Backward induction
    for j in range(n-1, -1, -1):
        for i in range(j+1):
            asset_prices[i] = S * (u ** (j-i)) * (d ** i)
            option_prices[i] = np.exp(-r * dt) * (p * option_prices[i] + (1 - p) * option_prices[i+1])

    return option_prices[0]

# Futures price function
def futures_price(S, r, T):
    return S * np.exp(r * T)

# Load your Parquet file
df = pd.read_parquet('/Users/banti/Downloads/nifty/NIFTY_2024-02-20.parquet')

# Convert columns to datetime if not already
df['ExpiryDate'] = pd.to_datetime(df['ExpiryDate'])
df['p_date'] = pd.to_datetime(df['p_date'])

# Define parameters
r = 0.06
sigma = 0.2
n = 10

# Apply calculations
def calculate_prices(row):
    S = row['LTP']
    K = row['StrikePrice']
    T = calculate_days_to_maturity(row['ExpiryDate'], row['p_date'])
    if row['OptionType'] == 'PE':
        price = binomial_option_pricing(S, K, T, r, sigma, n, option_type='PE')
        return price
    elif row['OptionType'] == 'FUT':
        price = futures_price(S, r, T)
        return price
    else:
        return np.nan

df['PriceCalculated'] = df.apply(calculate_prices, axis=1)

# Save the results to a new Parquet file
df.to_parquet('/Users/banti/Downloads/nifty/NIFTY_2024-02-20.parquet', index=False)

# Print first few rows to verify
print(df.head())
