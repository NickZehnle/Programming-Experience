### Geometric Brownian Motion

#### Code

~~~python
import numpy as np
import matplotlib.pyplot as plt

mu = 0.1 #drift coeff
n = 100 #number of steps
T = 1 #time in years
M = 100 #num of sims
S0 = 100 #initial stock price
sigma = 0.3 #volatility 

dt = T/n #calc each time step

#sim using numpy arrays
St = np.exp(
    (mu - sigma ** 2 / 2) * dt
    + sigma * np.random.normal(0, np.sqrt(dt), size=(M,n)).T
)

#calc array of 1's
St = np.vstack([np.ones(M), St])

#multiply through by S0 and return cumulative product of elements along a given sim path (axis=0)
St = S0 * St.cumprod(axis=0)

#define time interval 
time = np.linspace(0,T,n+1)

#numpy array that is the same shape as St
tt = np.full(shape=(M,n+1), fill_value=time).T

plt.plot(tt,St)
plt.xlabel('Years $(t)$')
plt.ylabel('Stock price $(S_t)$')
plt.title('GBM Sim\n $dS_t = \mu S_t dt + \sigma S_t dW_t$\n $S_0 = 100, \mu = 0.1, \sigma = 0.3$') #fix value grabs
plt.show()
~~~

#### Description
I followed a tutorial on how to construct an elementary geometric brownian motion simulation. 

#### Output
As one can see from the code, I set S = 
