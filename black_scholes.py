import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from sklearn.metrics import r2_score
from datetime import datetime

# Load the dataset
file_path = "/Users/priyaanksheth/Downloads/Code/Finsearch/nifty/NIFTY_2024-02-21.parquet"
data = pd.read_parquet(file_path)
# fut_data = data.copy()
fut_data = data[data['OptionType'] == "FUT"]
data = data[data['OptionType'] != "FUT"]
data = data[:1000]
# Function to calculate d1 and d2
def calculate_d1_d2(S, K, T, r, sigma):
    # print("calculate_d1_d2 called")
    try:
        d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    except Exception as e:
        print("dtype(S): ", type(S))
        print("S is ", S)
        print("dtype(K): ", type(K))
        print("dtype(T): ", type(T))
        print("dtype(r): ", type(r))
        print("dtype(sigma): ", type(sigma))
        # print(e)
        # d1 = np.nan
        print(e)
    # print("d1 calculated" , d1)
    d2 = d1 - sigma * np.sqrt(T)
    # print("d2 calculated" , d2)
    # print("returning d1 and d2")
    return d1, d2

# Black-Scholes model for European call and put options
def black_scholes(S, K, T, r, sigma, option_type='C'):
    # print("black_scholes called")
    d1, d2 = calculate_d1_d2(S, K, T, r, sigma)
    # print("reaches here")
    if option_type == 'C':  # Call option
        # print("Call option")
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'P':  # Put option
        # print("Put option")
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

# Function to calculate implied volatility using Brent's method
def implied_volatility(S, K, T, r, market_price, option_type='C'):
    # print("implied_volatility called")
    def objective_function(sigma):
        # print("objective_function called")
        return black_scholes(S, K, T, r, sigma, option_type) - market_price

    implied_vol = brentq(objective_function, 1e-5, 5)
    # print(f"Implied volatility: {implied_vol}")
    return implied_vol

# Prepare the data
data['TickerDateTime'] = pd.to_datetime(data['TickerDateTime'])
data['ExpiryDate'] = pd.to_datetime(data['ExpiryDate'])
data['TimeToMaturity'] = (data['ExpiryDate'] - data['TickerDateTime']).dt.total_seconds() / (365 * 24 * 3600)

# Risk-free rate (can be updated based on current rate)
risk_free_rate = 0.05

# Calculate implied volatility and Black-Scholes premium
calculated_premiums = []
implied_volatilities = []

for index, row in data.iterrows():
    # print(index, "/", len(data))
    # S = row['Underlying']
    timestamp = row['TickerDateTime']
    fut_ltp = fut_data[fut_data['TickerDateTime'] == timestamp]['LTP'].values[0]
    # print("fut_ltp is ", fut_ltp)
    S = fut_ltp
    K = row['StrikePrice']
    T = row['TimeToMaturity']
    r = risk_free_rate
    market_price = row['LTP']
    option_type = 'C' if row['OptionType'] == 'CE' else 'P'
    
    try:
        iv = implied_volatility(S, K, T, r, market_price, option_type)
        implied_volatilities.append(iv)
        
        premium = black_scholes(S, K, T, r, iv, option_type)
        calculated_premiums.append(premium*0.65)
    except Exception as e:
        implied_volatilities.append(np.nan)
        calculated_premiums.append(np.nan)

data['ImpliedVolatility'] = implied_volatilities
data['CalculatedPremium'] = calculated_premiums

# Drop any rows with NaN values (for valid R2 calculation)
valid_data = data.dropna(subset=['CalculatedPremium', 'LTP'])

# Calculate R^2
r2 = r2_score(valid_data['LTP'], valid_data['CalculatedPremium'])

print(f"R^2 between the calculated premium and LTP: {r2}")

# Save the resulting data with calculated premiums and implied volatilities
output_path = "/Users/priyaanksheth/Downloads/Code/Finsearch/nifty/NIFTY_2024-02-21_with_IV_and_Premium.parquet"
valid_data.to_parquet(output_path)

print(f"Data with calculated premiums and implied volatilities saved to: {output_path}")