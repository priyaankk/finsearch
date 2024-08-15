import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from sklearn.metrics import r2_score

# Load the dataset
file_path = "/Users/priyaanksheth/Downloads/Code/Finsearch/nifty/NIFTY_2024-02-21.parquet"
data = pd.read_parquet(file_path)
fut_data = data[data['OptionType'] == "FUT"]
data = data[data['OptionType'] != "FUT"]
data = data[:5000]
# Function to calculate d1 and d2
def calculate_d1_d2(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

# Black-Scholes model for European call and put options
def black_scholes(S, K, T, r, sigma, option_type='C'):
    d1, d2 = calculate_d1_d2(S, K, T, r, sigma)
    if option_type == 'C':  # Call option
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'P':  # Put option
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

# Function to calculate implied volatility using Brent's method
def implied_volatility(S, K, T, r, market_price, option_type='C'):
    def objective_function(sigma):
        return black_scholes(S, K, T, r, sigma, option_type) - market_price

    try:
        implied_vol = brentq(objective_function, 1e-5, 5)
    except ValueError:
        implied_vol = np.nan  # Handle cases where Brent's method fails to converge

    return implied_vol

# Function for Monte Carlo simulation for European option pricing
def monte_carlo_option_pricing(S, K, T, r, sigma, option_type='C', num_simulations=10000, num_steps=252):
    dt = T / num_steps
    price_paths = np.zeros((num_simulations, num_steps + 1))
    price_paths[:, 0] = S
    
    for t in range(1, num_steps + 1):
        z = np.random.standard_normal(num_simulations)
        price_paths[:, t] = price_paths[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    
    if option_type == 'C':
        payoffs = np.maximum(price_paths[:, -1] - K, 0)
    elif option_type == 'P':
        payoffs = np.maximum(K - price_paths[:, -1], 0)
    
    discounted_payoff = np.exp(-r * T) * payoffs
    option_price = np.mean(discounted_payoff)
    
    return option_price

# Prepare the data
data['TickerDateTime'] = pd.to_datetime(data['TickerDateTime'])
data['ExpiryDate'] = pd.to_datetime(data['ExpiryDate'])
data['TimeToMaturity'] = (data['ExpiryDate'] - data['TickerDateTime']).dt.total_seconds() / (365 * 24 * 3600)

# Risk-free rate (can be updated based on current rate)
risk_free_rate = 0.05

# Calculate Implied Volatility and Monte Carlo premiums
implied_volatilities = []
monte_carlo_premiums = []

for index, row in data.iterrows():
    print(index, "/", len(data))
    timestamp = row['TickerDateTime']
    fut_ltp = fut_data[fut_data['TickerDateTime'] == timestamp]['LTP'].values[0]
    S = fut_ltp
    K = row['StrikePrice']
    T = row['TimeToMaturity']
    r = risk_free_rate
    market_price = row['LTP']
    option_type = 'C' if row['OptionType'] == 'CE' else 'P'
    
    try:
        # Calculate Implied Volatility
        iv = implied_volatility(S, K, T, r, market_price, option_type)
        implied_volatilities.append(iv)
        
        # Use Monte Carlo simulation to calculate premium
        mc_premium = monte_carlo_option_pricing(S, K, T, r, iv, option_type)
        monte_carlo_premiums.append(mc_premium)
    except Exception as e:
        implied_volatilities.append(np.nan)
        monte_carlo_premiums.append(np.nan)

data['ImpliedVolatility'] = implied_volatilities
data['MonteCarloPremium'] = monte_carlo_premiums

# Drop any rows with NaN values (for valid R2 calculation)
valid_data_mc = data.dropna(subset=['MonteCarloPremium', 'LTP'])

# Calculate R^2 for Monte Carlo method
r2_mc = r2_score(valid_data_mc['LTP'], valid_data_mc['MonteCarloPremium'])

print(f"R^2 between the Monte Carlo premium and LTP: {r2_mc}")

# Save the resulting data with Monte Carlo premiums
output_path_mc = "/Users/priyaanksheth/Downloads/Code/Finsearch/nifty/NIFTY_2024-02-21_with_IV_and_MC_Premium.parquet"
valid_data_mc.to_parquet(output_path_mc)

print(f"Data with calculated premiums and implied volatilities saved to: {output_path_mc}")
