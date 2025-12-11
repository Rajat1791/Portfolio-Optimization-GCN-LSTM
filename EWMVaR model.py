import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
# Set the start and end dates for fetching the historical data
start_date = '2015-01-01'
end_date = '2025-07-01'

# Define the list of assets
assets = ["BHARTIARTL.NS", "JSWSTEEL.NS", "RELIANCE.NS", "TATAMOTORS.NS", "SUNPHARMA.NS", "COALINDIA.NS", "CIPLA.NS", "TECHM.NS", "DRREDDY.NS", "POWERGRID.NS"]

# Fetch the historical data for the assets
data = yf.download(assets, start=start_date, end=end_date)['Close']
print(data)
def preprocess_data(data):
    """
    Preprocess the data DataFrame.
    1. Ensure the index is sorted (assuming the index is dates).
    2. Handle missing values by forward filling and then dropping any remaining missing values.
    3. If the data appears to be prices (values mostly > 1), compute daily returns.
       Otherwise, assume the data is already in returns.

    Parameters:
    - data: pandas DataFrame containing historical data.

    Returns:
    - returns: DataFrame of daily returns.
    """
    # Ensure the index is sorted
    data = data.sort_index()

    # Forward fill missing values and then drop any remaining NaNs
    data = data.fillna(method='ffill').dropna()

    # Check if data appears to be prices (values > 1) or returns (typically small numbers)
    # This is a heuristic: if the mean of the data is greater than 1, assume prices.
    if data.mean().mean() > 1:
        print("Data appears to be prices. Converting to daily returns.")
        returns = data.pct_change().dropna()
    else:
        print("Data appears to be daily returns. Proceeding without conversion.")
        returns = data.copy()

    return returns
# Preprocess the downloaded data to get daily returns
returns_data = preprocess_data(data)

# You can check a sample of the processed returns:
print("Processed Daily Returns:")
print(returns_data.head())
# Helper Functions
# -----------------------------
def exponential_weights(n, halflife):
    """
    Compute exponential weights for n observations with given halflife.
    """
    tau = halflife / np.log(2)
    weights = np.exp(-np.arange(n) / tau)
    return weights / np.sum(weights)

def weighted_quantile(values, quantile, sample_weight):
    """
    Compute the weighted quantile of a 1D numpy array.
    """
    sorter = np.argsort(values)
    values_sorted = values[sorter]
    weights_sorted = sample_weight[sorter]
    cumsum = np.cumsum(weights_sorted)
    cutoff = quantile * cumsum[-1]
    return values_sorted[np.searchsorted(cumsum, cutoff)]
# Objective Function
# -----------------------------
def objective(x, returns_data, halflife, risk_aversion, alpha):
    """
    Given portfolio weights x, compute the exponentially weighted portfolio return distribution,
    then compute:
      weighted_mean = sum(exp_weights * portfolio_returns)
      weighted_VaR = - weighted quantile at level 'alpha' of portfolio returns
    and define the objective to maximize:
      f = weighted_mean - risk_aversion * weighted_VaR
    We return -f because we will use a minimizer.
    """
    port_returns = returns_data.dot(x)  # daily portfolio returns
    n = len(port_returns)
    exp_w = exponential_weights(n, halflife)
    weighted_mean = np.sum(exp_w * port_returns)
    quant = weighted_quantile(port_returns.values, alpha, exp_w)
    weighted_VaR = -quant  # making VaR positive (i.e. higher VaR means more risk)
    f = weighted_mean - risk_aversion * weighted_VaR
    return -f
# Portfolio Optimization Function
# -----------------------------
def optimize_portfolio(returns_data, halflife=60, risk_aversion=1.0, alpha=0.05):
    """
    Optimize the portfolio weights for the exponentially weighted mean-VaR model.

    Parameters:
      returns_data  : pandas DataFrame of daily returns (with a DatetimeIndex)
      halflife      : halflife parameter for the exponential weighting (in days)
      risk_aversion : weight for the VaR penalty in the objective function
      alpha         : significance level for VaR (e.g., 0.05 for 5% VaR)

    Returns:
      optimal_weights: optimized portfolio weights (numpy array)
      optimal_obj    : the maximized objective value (note: the function returns the negative value, so we invert it)
    """
    num_assets = returns_data.shape[1]
    x0 = np.ones(num_assets) / num_assets  # initial guess: equal weights
    bounds = [(0, 1)] * num_assets
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    result = minimize(objective, x0, args=(returns_data, halflife, risk_aversion, alpha),
                      method='SLSQP', bounds=bounds, constraints=cons)

    optimal_weights = result.x
    optimal_obj = -result.fun  # invert to get the maximized value
    return optimal_weights, optimal_obj

if __name__ == "__main__":
    # Load your returns data (assumed to be 10 years of daily returns with a proper DatetimeIndex)
    # Example:
    # returns_data = pd.read_csv('your_data_file.csv', index_col='Date', parse_dates=True)
    # For this example, ensure that "returns_data" is defined.

    optimal_weights, optimal_obj = optimize_portfolio(returns_data, halflife=60, risk_aversion=1.0, alpha=0.05)

    print("Optimal Portfolio Weights (Exp. Weighted Mean-VaR):")
    print(optimal_weights)
    print("Maximized Objective Value:", optimal_obj)

    # Use 'returns_data' (the processed returns) here, not 'returns'
    ann_ret, ann_vol, ann_VaR, sharpe = compute_portfolio_annual_performance(optimal_weights, returns_data, alpha=0.05, risk_free_rate=0.02)

    print("\nPortfolio Annual Performance Metrics (Exp. Weighted Mean-VaR):")
    print(f"Annualized Return: {ann_ret:.4f}")
    print(f"Annualized Volatility: {ann_vol:.4f}")
    print(f"Annualized VaR: {ann_VaR:.4f}")
    print(f"Sharpe Ratio: {sharpe:.4f}")
