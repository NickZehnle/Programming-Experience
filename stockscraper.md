## Implied Volatility Tracking Web Scraper

#### Brief Summary
I designed a web scraper that gathers the market price of the underlying asset, the market price of the call option, and the risk-free rate in real time. The user inputs the stock, the strike price, and the time to maturity that is of interest to them. Then, the implementation of the Newton-Raphson method (with some improvements -- see later on) calculates the implied volatility at that instant. Furthermore, vega, volga, and ultima are calculated. The program runs according to a specified time interval until the market closes. Nonetheless, it will produce a single row of data after close if the user wishes to check their current position after the trading day. All of the data is continuously placed into a dataframe that uploads to an Excel file. 

#### Code

https://github.com/NickZehnle/Programming-Experience/blob/8a0b8a7fa28fcb51406a2484af39e1b80c6f1cfb/stockscraper.py#L1-L192

#### Functionality
Below is an image of the terminal during runtime. It first collects the user input for stock, strike price, and days until expiration. Then, it scrapes the market price of the underlying asset. In order to find the market price of the option it begins at the top of the straddle and iterates through the strike prices until it finds the row containing the strike price input by the user. The program subsequently moves over the the Last Price column and scrapes the option price. Lastly, it scrapes the risk-free rate, calculates, and appends the findings to the dataframe.

#### Data Examples
Note the path label in the code of the program above. I have it so the Excel file is marked by the date followed by the name of the stock.

#### Upcoming Improvements
As aforementioned, the Black-Scholes formula and its derivatives with respect to implied volatility used in the program do not take into account dividend payoffs. Ergo, an easy update would be to scrape the dividend payoffs of the stock and substitute the formulas that include a measure for dividends. 
The program also only considers call options. An obvious expansion would be to insert the Black-Scholes formula for put options, update the derivatives with respect to volatility to encompass their put option alternatives, and augment functions such as get_price and impvol_calc. 
I do not intend to make the transition from European option to American option calculations as I was primarily focused on integrating the Newton-Raphson method with the functional improvements brought about through Taylor Series expansion. That said, I plan to work on a binomial pricing model for American options in the future.
