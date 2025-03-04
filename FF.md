## Polynomial Fama-French Portfolio Construction
#### Results
[Portfolio Performance Report](FF_Report_DIA2018-25.pdf)

#### Code
~~~python
from AlgorithmImports import *
from io import StringIO
import pandas as pd
import numpy as np
import cvxpy as cp
import time

class FactorPortfolioAlgorithm(QCAlgorithm):
    
    def initialize(self):
        self.set_start_date(2018, 1, 1)
        self.set_end_date(2025, 1, 1)
        self.set_cash(1_000_000)
        self.settings.minimum_order_margin_portfolio_percentage = 0

        self._lookback = self.get_parameter("lookback_days", 6300) # 25 years
        self._universe_size = self.get_parameter("universe_size", 50)

        self.SetBenchmark("DIA")

        schedule_symbol = Symbol.create("DIA", SecurityType.EQUITY, Market.USA)
        date_rule = self.date_rules.month_start(schedule_symbol)
        self.universe_settings.schedule.on(date_rule)
        self._universe = self.add_universe(self._select_assets)

        file = self.download('https://www.dropbox.com/scl/fi/x2aer7ucoygth1sdst0c1/factors.csv?rlkey=hww1z3vqx0z0dsdvvpdxbz2l1&st=zsqgmrgh&dl=1')
        self.factors = pd.read_csv(StringIO(file))

        self.schedule.on(
            date_rule, 
            self.time_rules.after_market_open(schedule_symbol, 1), 
            self._trade
        )

        chart = Chart("No. Assets")
        chart.AddSeries(Series("Holdings", SeriesType.Line, 0))
        self.AddChart(chart)

        self.history_cache = {}

    def _select_assets(self, fundamental):
        ''' Select securities in Dow Jones market cap range that have the most $ volume'''
        return [
            f.symbol 
            for f in sorted(
                [f for f in fundamental if f.price > 2 and 30e9 <= f.market_cap <= 2.5e12], 
                key=lambda f: f.dollar_volume
            )[-self._universe_size:]
        ]

    def _trade(self):
        '''Get tradeable assets in universe, compute weights, and execute trades'''
        start_time = time.time()

        tradeable_assets = [
            symbol 
            for symbol in self._universe.selected 
            if (self.securities[symbol].price and 
            symbol in self.current_slice.quote_bars)
        ]

        cached_history = self._get_cached_history(tradeable_assets)
        history = pd.DataFrame(cached_history)
        monthly_history = history.resample('M').last() 

        elapsed_time = time.time() - start_time
        self.Debug(f"History took {elapsed_time:.2f} seconds.")

        start_time = time.time()

        weights = self._get_weights(monthly_history, tradeable_assets)

        elapsed_time = time.time() - start_time
        self.Debug(f"_get_weights took {elapsed_time:.2f} seconds.")

        start_time = time.time()

        self.set_holdings(
            [
                PortfolioTarget(symbol, weight) 
                for symbol, weight in weights.items()
            ], 
            True
        )

        elapsed_time = time.time() - start_time
        self.Debug(f"_set_holdings took {elapsed_time:.2f} seconds.")

        total_stocks = sum(1 for security in self.Portfolio.Values if security.Invested)
        self.Plot("No. Assets", "Holdings", total_stocks)

        del history

    def _get_cached_history(self, tradeable_assets):
        ''' Store history from previous timestep for faster execution'''
        cached_assets = list(self.history_cache.keys())
        for symbol in cached_assets:
            if symbol not in tradeable_assets:
                del self.history_cache[symbol]

        existing_assets = [symbol for symbol in tradeable_assets if symbol in self.history_cache]
        if existing_assets:
            latest_data = self.history(existing_assets, 1, Resolution.DAILY).close.unstack(level=0)
            for symbol in existing_assets:
                self.history_cache[symbol] = (pd.concat([self.history_cache[symbol], latest_data[symbol]]))

        new_assets = [symbol for symbol in tradeable_assets if symbol not in self.history_cache]
        if new_assets:
            history_data = self.history(new_assets, self._lookback, Resolution.DAILY).close.unstack(level=0)
            for symbol in history_data.columns:
                self.history_cache[symbol] = history_data[symbol] 

        return self.history_cache

    def _get_weights(self, history, tradeable_assets):
        '''Compute weights based on target betas''' 
        valid_assets = [
            symbol for symbol in tradeable_assets 
            if len(history[symbol].dropna()) >= 240
        ] # 20 years of data

        returns = pd.DataFrame({
            str(symbol): history[symbol].pct_change() * 100
            for symbol in valid_assets
        }).dropna()
    
        returns['Date'] = returns.index.to_period('M').astype(str)

        df = pd.merge(returns, self.factors, on='Date')
    
        X = df[['LT_Rev', 'Mom', 'Mkt', 'SMB', 'HML', 'SMB_CMA', 'Mkt2']].values
        Y = df.iloc[:, :-8].values

        betas = np.linalg.inv(X.T @ X) @ X.T @ Y
        betas = betas.T 

        target = np.array([ 0.389, -0.147,  0.828,  0.887,  0.315, 0.076, -0.016])

        num_assets = betas.shape[0]
        weights = cp.Variable(num_assets)

        objective = cp.Minimize(cp.sum_squares(betas.T @ weights - target))

        constraints = [
            cp.sum(weights) == 1,  # Weights sum to 1
            weights >= .005, # Short-selling not allowed
            weights <= .2 # Make sure no one stock is too dominant
        ]

        problem = cp.Problem(objective, constraints)

        start_time = time.time()

        problem.solve(solver=cp.OSQP, max_iter=5000)

        elapsed_time = time.time() - start_time
        self.Debug(f"Optimization took {elapsed_time:.2f} seconds.")
        
        computed_weights = {
            valid_assets[i]: weights.value[i] for i in range(num_assets)
        }

        return computed_weights
~~~

#### Research
~~~python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import datetime as dt
from statsmodels.formula.api import ols
from statsmodels.api import qqplot
from sklearn.model_selection import train_test_split
import pandas_datareader.data as web
from pandas_datareader.famafrench import get_available_datasets
from statsmodels.stats.anova import anova_lm

import warnings
warnings.filterwarnings("ignore")

available_data = get_available_datasets()
#available_data

start  = '1970-01-01'
end ='2024-11-27'
factors = web.DataReader('F-F_Research_Data_5_Factors_2x3','famafrench', start=start ,end=end)[0]
ports = web.DataReader('100_Portfolios_10x10','famafrench', start=start ,end=end)[0]
ports = ports.sub(factors.RF, axis=0)
ports.columns = ports.columns.str.replace(' ', '_')
#print(factors)
#print(ports.columns)

mom = web.DataReader('F-F_Momentum_Factor','famafrench', start=start ,end=end)[0]
st_rev = web.DataReader('F-F_ST_Reversal_Factor','famafrench', start=start ,end=end)[0]
lt_rev = web.DataReader('F-F_LT_Reversal_Factor','famafrench', start=start ,end=end)[0]
factors = pd.merge(mom, factors, on='Date')
factors = pd.merge(st_rev, factors, on='Date')
factors = pd.merge(lt_rev, factors, on='Date')
factors.drop(['RF'], axis=1, inplace=True)
factors.rename(columns={'Mom   ':'Mom', 'Mkt-RF':'Mkt'}, inplace=True)
#print(factors)

for col1, col2 in itertools.combinations(factors.columns[2:], 2):
    factors[f'{col1}_{col2}'] = factors[col1] * factors[col2]

for col in factors.columns[:8]:
    factors[f'{col}2']=factors[col]**2
    factors[f'{col}3']=factors[col]**3

#print(factors)

def back_elim(model, F_cutoff):
    ols_model = ols(model, df).fit()
    anova_results = anova_lm(ols_model,typ=2)
    F = anova_results.F
    while (np.min(F) < F_cutoff):
        if '+' + F.idxmin() == model[-(len(F.idxmin())+1):]:
            drop_end = '+' + F.idxmin()
            model = model.replace(drop_end, '')
        drop_start = str('~') + F.idxmin() + str('+')
        model = model.replace(drop_start, '~')
        drop_mid = str('+') + F.idxmin() + str('+')
        model = model.replace(drop_mid, '+')
        ols_model = ols(model, df).fit()
        anova_results=anova_lm(ols_model,typ=2)
        F = anova_results.F
    return model

string = str('~')
for factor in factors.columns:
    string += factor+str('+')
string = string[0:-1]

ports_1970, strings_1970, alphas_1970 = [], [], []

for port in ports.columns:
    df = pd.merge(ports.loc[:,port], factors, on='Date')
    new_string = f'{port}' + string 
    reduced_model = back_elim(new_string, 10)
    model=ols(reduced_model, df).fit()
    if model.pvalues[0]<.05:
        ports_1970.append(port)
        strings_1970.append(reduced_model)
        alphas_1970.append(model.params[0])
        
print(ports_1970)
print(alphas_1970)

ports_1980, strings_1980, alphas_1980 = [], [], []

for port in ports.columns:
    df = pd.merge(ports.loc['1980-01':,port], factors, on='Date')
    new_string = f'{port}' + string 
    reduced_model = back_elim(new_string, 10)
    model=ols(reduced_model, df).fit()
    if model.pvalues[0]<.05:
        ports_1980.append(port)
        strings_1980.append(reduced_model)
        alphas_1980.append(model.params[0])
        
print(ports_1980)
print(alphas_1980)

ports_1990, strings_1990, alphas_1990 = [], [], []

for port in ports.columns:
    df = pd.merge(ports.loc['1990-01':,port], factors, on='Date')
    new_string = f'{port}' + string 
    reduced_model = back_elim(new_string, 10)
    model=ols(reduced_model, df).fit()
    if model.pvalues[0]<.05:
        ports_1990.append(port)
        strings_1990.append(reduced_model)
        alphas_1990.append(model.params[0])
        
print(ports_1990)
print(alphas_1990)

sig_ports = list(set(ports_1970) & set(ports_1980) & set(ports_1990))
print(sig_ports)

#1970
for p in sig_ports:
    df = pd.merge(ports.loc[:,p], factors, on='Date')
    model = ols(strings_1970[ports_1970.index(p)], df).fit()
    print(model.summary())

#1980
for p in sig_ports:
    df = pd.merge(ports.loc['1980-01':,p], factors, on='Date')
    model = ols(strings_1980[ports_1980.index(p)], df).fit()
    print(model.summary())

#1990
for p in sig_ports:
    df = pd.merge(ports.loc['1990-01':,p], factors, on='Date')
    model = ols(strings_1990[ports_1990.index(p)], df).fit()
    print(model.summary())

load_df = pd.merge(ports.loc['1990-01':,'SMALL_HiBM'], factors, on='Date')
load_model = ols(strings_1990[ports_1990.index('SMALL_HiBM')], load_df).fit()
load_model.summary()

np.array(load_model.params[1:])

load_factors = factors[['LT_Rev', 'Mom', 'Mkt', 'SMB', 'HML', 'SMB_CMA', 'Mkt2']]
#load_factors.to_csv('factors.csv',index=True)
~~~




