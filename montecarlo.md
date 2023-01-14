### Monte Carlo, VaR, and CVaR

#### Code 

~~~python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from pandas_datareader import data as pdr
import yfinance as yf

yf.pdr_override()

def get_data(stocks, start, end):
    stockData = pdr.get_data_yahoo(stocks, start, end)
    stockData = stockData['Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix
~~~

~~~python
stockList = ['AMZN', 'NVDA', 'TSLA', 'GM']
stocks = [stock for stock in stockList]
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=300)

meanReturns, covMatrix = get_data(stocks, startDate, endDate)

weights = np.random.random(len(meanReturns))
weights /= np.sum(weights)
~~~

~~~python
#montecarlo
mc_sims = 100 #num simulations
T = 100 #timeframe in days

meanM = np.full(shape= (T, len(weights)), fill_value=meanReturns)
meanM = meanM.T #transpose

portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)

initialPortfolio = 10000

for m in range(0, mc_sims):
    #using Cholesky decomp
    Z = np.random.normal(size=(T, len(weights)))
    L = np.linalg.cholesky(covMatrix) #works out lower triangle
    dailyReturns = meanM + np.inner(L, Z)
    portfolio_sims[:,m] = np.cumprod(np.inner(weights, dailyReturns.T)+1)*initialPortfolio

plt.plot(portfolio_sims)
plt.ylabel('Portfolio Value ($)')
plt.xlabel('Days')
plt.title('MC simulation of a stock portfolio')
~~~

~~~python
def mcVaR(returns, alpha=5):
    if isinstance(returns, pd.Series):
        return np.percentile(returns, alpha)
    else:
        raise TypeError('Expected pandas data series')
    
def mcCVaR(returns, alpha=5):
    if isinstance(returns, pd.Series):
        belowVaR = returns <= mcVaR(returns, alpha=alpha)
        return returns[belowVaR].mean()
    else:
        raise TypeError('Expected pandas data series')
    
portResults = pd.Series(portfolio_sims[-1,:])

VaR = initialPortfolio - mcVaR(portResults, alpha=5)
CVaR = initialPortfolio - mcCVaR(portResults, alpha=5)

print('VaR ${}'.format(round(VaR,3)))
print('CVaR ${}'.format(round(CVaR,3)))
plt.show()
~~~

#### Description
I followed a tutorial of how to implement a monte carlo simulation that predicts the performance of a portfolio. Additionally, I learned how to calculate VaR (Value at Risk) and CVaR (Conditional Value at Risk).

#### Output

![](mc-ex.png)
