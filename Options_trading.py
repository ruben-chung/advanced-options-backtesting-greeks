import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# Black-Scholes calculation
def calculate_greeks(S, K, T, r, sigma, option_type='C'):
    if T <= 0:
        return np.nan, np.nan, np.nan, np.nan
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type.upper() in ['C', 'CALL']:
        delta = norm.cdf(d1)
    elif option_type.upper() in ['P', 'PUT']:
        delta = norm.cdf(d1) - 1
    else:
        raise ValueError("Invalid option_type. Use 'C'/'CALL' or 'P'/'PUT'.")
    
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    vega = S * norm.pdf(d1) * np.sqrt(T)
    
    return delta, gamma, theta, vega


data2 = pd.read_csv('NVDA_excel.csv', parse_dates=['Date']) # Can change this to other data
data = data2[data2["expiry_days"] > 0].copy()
data = data.sort_values(by='Date').reset_index(drop=True)
data['T'] = data['expiry_days'] / 365

data[['Delta', 'Gamma', 'Theta', 'Vega']] = data.apply(
    lambda row: calculate_greeks(
        S=row['adjusted close'],
        K=row['Strike'],
        T=row['T'],
        r=0.01,
        sigma=row['iv'],
        option_type=row['option_type']  
    ) if row['T'] > 0 else (np.nan, np.nan, np.nan, np.nan),
    axis=1,
    result_type='expand'
)

data.dropna(subset=['Delta', 'Gamma', 'Theta', 'Vega'], inplace=True)
data = data[data['mean price'] > 0].copy()
iv_mean = data['iv'].mean()


buy_call = (data['option_type'] == 'C') & (data['iv'] < iv_mean) & (data['Delta'] > 0.5)
sell_call = (data['option_type'] == 'C') & ((data['iv'] >= iv_mean) | (data['Delta'] <= 0.5))

buy_put = (data['option_type'] == 'P') & (data['iv'] < iv_mean) & (data['Delta'] < -0.5)
sell_put = (data['option_type'] == 'P') & ((data['iv'] >= iv_mean) | (data['Delta'] >= -0.5))

data['Signal'] = 0
data.loc[buy_call | buy_put, 'Signal'] = 1
data.loc[sell_call | sell_put, 'Signal'] = -1
data['Signal'] = data['Signal'].diff() # buy signal is 2 and sell is -2
data.reset_index(drop=True, inplace=True)

# How much money would be made from a inital 10,000$ investment
portfolio = {'capital': 10000, 'holdings': 0, 'total_value': 10000}
trade_log = []
position_open = False  
portfolio_values = []

for i in range(len(data)):
    signal = data['Signal'].iloc[i]
    price = data['mean price'].iloc[i]  
    date = data['Date'].iloc[i]
    
    # Buy signal
    if signal == 2 and not position_open and portfolio['capital'] > price:
        num_options = int(portfolio['capital'] // price)
        if num_options > 0:
            cost = num_options * price
            portfolio['capital'] -= cost
            portfolio['holdings'] += num_options
            position_open = True
            trade_log.append({'Date': date, 'Action': 'Buy', 'Price': price, 'Options': num_options})
            print(f"Executed Buy: {num_options} options at ${price:.2f} on {date.date()}")
    
    # Sell signal
    elif signal == -2 and position_open and portfolio['holdings'] > 0:
        revenue = portfolio['holdings'] * price
        portfolio['capital'] += revenue
        portfolio['holdings'] = 0
        position_open = False
        trade_log.append({'Date': date, 'Action': 'Sell', 'Price': price, 'Options': 0})
        print(f"Executed Sell: All options at ${price:.2f} on {date.date()}")
    
    portfolio['total_value'] = portfolio['capital'] + (portfolio['holdings'] * price)
    portfolio_values.append(portfolio['total_value'])


data['Portfolio_Value'] = portfolio_values
trade_log = pd.DataFrame(trade_log)

print(f"\nFinal Portfolio Value: ${portfolio['total_value']:.2f}")
print("\nTrade Log:")
print(trade_log)

if trade_log.empty:
    print("\nNo trades executed.")
else:
    plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], data['adjusted close'], label='Option Price', alpha=0.7)  #adj close chnaged from close
    plt.scatter(trade_log[trade_log['Action'] == 'Buy']['Date'], 
                trade_log[trade_log['Action'] == 'Buy']['Price'], 
                label='Buy Signal', marker='^', color='green')
    plt.scatter(trade_log[trade_log['Action'] == 'Sell']['Date'], 
                trade_log[trade_log['Action'] == 'Sell']['Price'], 
                label='Sell Signal', marker='v', color='red')
    plt.legend()
    plt.title('Options Backtesting Strategy with Greeks')
    plt.xlabel('Date')
    plt.ylabel('Option Price')
    plt.grid()
    plt.show()


initial_capital = 10000
final_capital = portfolio['total_value']
total_return = (final_capital - initial_capital) / initial_capital * 100

print(f"\nPerformance Metrics:")
print(f"Initial Capital: ${initial_capital:.2f}")
print(f"Final Portfolio Value: ${final_capital:.2f}")
print(f"Total Return: {total_return:.2f}%")
