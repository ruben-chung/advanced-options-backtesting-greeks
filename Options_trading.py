print("hello")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# Black-Scholes Greeks Calculator
def calculate_greeks(S, K, T, r, sigma, option_type='C'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    delta = norm.cdf(d1) if option_type == 'call' else -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    vega = S * norm.pdf(d1) * np.sqrt(T)
    
    return delta, gamma, theta, vega

# Load Data
data2 = pd.read_csv('NVDA_excel.csv') #parse_dates=['date'])
data = data2[data2["expiry_days"] > 0]

print("data", data)
print("data2", data2)


# Calculate Greeks and Implied Volatility
data['Delta'], data['Gamma'], data['Theta'], data['Vega'] = zip(
    *data.apply(lambda row: calculate_greeks(
        S=row['adjusted close'], #idk to use adjusted or not, was underlying price before
        K=row['Strike'], 
        T=row['expiry_days'], #in days
        r=0.01,  # Assume 1% risk-free rate
        sigma=row['iv'],
        option_type='C'
    ), axis=1)
)
print(data)
print("after delta gamma")

# # Define Trading Signals
# Long call when IV is below mean and Delta > 0.5
iv_mean = data['iv'].mean()
data['Signal'] = np.where((data['iv'] < iv_mean) & (data['Delta'] > 0.5), 1, 0)  #creating new column
data['Signal'] = data['Signal'].diff()  # 1 for buy, -1 for sell

data.to_csv('optionsiv.csv')

print("done")

# # Backtesting Logic
# portfolio = {'capital': 1000      0, 'holdings': 0, 'total_value': 10000}
# trade_log = []

# for i in range(1, len(data)):
#     signal = data['Signal'].iloc[i]
#     price = data['adjusted close'].iloc[i]  #changed from Close
    
#     # Buy signal
#     if signal == 1 and portfolio['capital'] > price:
#         num_options = int(portfolio['capital'] // price)
#         cost = num_options * price
#         portfolio['capital'] -= cost
#         portfolio['holdings'] += num_options
#         trade_log.append({'Date': data['Date'].iloc[i], 'Action': 'Buy', 'Price': price, 'Options': num_options})
    
#     # Sell signal
#     elif signal == -1 and portfolio['holdings'] > 0:
#         revenue = portfolio['holdings'] * price
#         portfolio['capital'] += revenue
#         portfolio['holdings'] = 0
#         trade_log.append({'Date': data['Date'].iloc[i], 'Action': 'Sell', 'Price': price, 'Options': 0})
    
#     # Update portfolio value
#     portfolio['total_value'] = portfolio['capital'] + (portfolio['holdings'] * price)

# # Convert Trade Log to DataFrame
# trade_log = pd.DataFrame(trade_log)

# # Plot Results
# plt.figure(figsize=(12, 6))
# plt.plot(data['Date'], data['adjusted close'], label='Option Price', alpha=0.7)  #adj close chnaged from close
# plt.scatter(trade_log[trade_log['Action'] == 'Buy']['Date'], 
#             trade_log[trade_log['Action'] == 'Buy']['Price'], 
#             label='Buy Signal', marker='^', color='green')
# plt.scatter(trade_log[trade_log['Action'] == 'Sell']['Date'], 
#             trade_log[trade_log['Action'] == 'Sell']['Price'], 
#             label='Sell Signal', marker='v', color='red')
# plt.legend()
# plt.title('Options Backtesting Strategy with Greeks')
# plt.xlabel('Date')
# plt.ylabel('Option Price')
# plt.grid()
# plt.show()

# # Print Performance
# print(f"Final Portfolio Value: ${portfolio['total_value']:.2f}")
# print(f"Trade Log:\n{trade_log}")

