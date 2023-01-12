# Implied Volatility Tracking Web Scraper

#### Brief Summary
I designed a web scraper that gathers the market price of the underlying asset, the market price of the call option, and the risk-free rate in real time. 
The user inputs the stock, the strike price, and the time to maturity that is of interest to them. Then, the implementation of the Newton-Raphson method 
(with some improvements -- see later on) calculates the implied volatility at that instant. Furthermore, vega, volga, and ultima are calculated. The program runs
according to a specified time interval until the market closes. Nonetheless, it will produce a single row of data after close if the user wishes to check their current position
after the trading day. All of the data is continuously placed into a dataframe that uploads to an Excel file. 

#### Code
https://github.com/NickZehnle/Programs/blob/7538619dd8c37ebac19932ceb161add8d12b4d87/stockscraper.py#L1-L192
#### How it functions
#### Data Examples
#### Improvements to be made
