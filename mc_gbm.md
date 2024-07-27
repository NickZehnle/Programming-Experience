## Monte Carlo VaR and CVaR (Geometric Brownian Motion)

#### Code
~~~python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from pandas_datareader import data as pdr
import yfinance as yf
import scipy.stats as scs

yf.pdr_override()

def get_data(stocks, start, end):
    stockData = pdr.get_data_yahoo(stocks, start, end)
    stockData = stockData['Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    sdReturns = returns.std()
    return meanReturns, sdReturns

stockList = ['AMZN', 'NVDA', 'AAPL', 'PFE', 'EQIX', 'LULU', 'CCI', 'VLO']
stocks = [stock for stock in stockList]
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=100)

meanReturns, sdReturns = get_data(stocks, startDate, endDate)

weights = np.random.random(len(meanReturns))
weights /= np.sum(weights)
print(weights)

mu = np.dot(weights, meanReturns)*np.sqrt(252)
sigma = np.dot(weights, sdReturns)*np.sqrt(252)
rfr = pdr.get_data_yahoo('^IRX', dt.datetime.now()-dt.timedelta(days=1), dt.datetime.now())
rfr=float(rfr['Adj Close'] / 100)
T = 7 
S0 = 10000
Dt=T/252
n_sims=100000

a = (rfr - .5 * sigma**2)*Dt
b = sigma * np.sqrt(Dt)
Z = np.random.normal(0,1,(n_sims,1))

sims = S0 * np.exp(a + b*Z)

mcReturns = sims - S0

alpha = [.01, .1, 1, 5, 10]

VaR = scs.scoreatpercentile(mcReturns, alpha)

def CVaR(returns):
    cvar = np.zeros(len(alpha))
    for i, var in enumerate(VaR):
        belowVaR = returns <= var
        cvar[i] = returns[belowVaR].mean()
    return cvar

cvar = CVaR(mcReturns)

df = pd.DataFrame(VaR, alpha, columns=['VaR'])
df['CVaR'] = cvar
print(df)

plt.hist(mcReturns, density=True, bins=500)

plt.axvline(df.loc[0.01]['VaR']+1, color='r', linestyle='dashed')
plt.axvline(df.loc[0.1]['VaR']+1, color='r', linestyle='dashed')
plt.axvline(df.loc[1]['VaR']+1, color='r', linestyle='dashed')
plt.axvline(df.loc[5]['VaR']+1, color='r', linestyle='dashed')
plt.axvline(df.loc[10]['VaR']+1, color='r', linestyle='dashed')
plt.axvline(df.loc[0.01]['CVaR']+1, color='g', linestyle='dashed')
plt.axvline(df.loc[0.1]['CVaR']+1, color='purple', linestyle='dashed')
plt.axvline(df.loc[1]['CVaR']+1, color='g', linestyle='dashed')
plt.axvline(df.loc[5]['CVaR']+1, color='purple', linestyle='dashed')
plt.axvline(df.loc[10]['CVaR']+1, color='g', linestyle='dashed')

plt.show()
~~~
