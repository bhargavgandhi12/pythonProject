"""
Sample data


                 Open       High        Low      Close  Adj Close    Volume
Date
2014-12-31  46.730000  47.439999  46.450001  46.450001  42.848763  21552500
2015-01-02  46.660000  47.419998  46.540001  46.759998  43.134731  27913900
2015-01-05  46.369999  46.730000  46.250000  46.330002  42.738068  39673900
2015-01-06  46.380001  46.750000  45.540001  45.650002  42.110783  36447900
2015-01-07  45.980000  46.459999  45.490002  46.230000  42.645817  29114100


"""


import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline

fb = pd.DataFrame.from_csv('../data/facebook.csv')
print(fb.head())
print(fb.shape)
print(fb.describe())
fb_2015 = fb.loc['2015-01-01':'2015-12-31']
print(fb_2015.loc['2015-03-16'])
print(fb.iloc[0, 0])
fb['PriceDiff'] = fb['Close'].shift(-1) - fb['Close']
fb['Return'] = fb['PriceDiff'] /fb['Close']
fb['Direction'] = [1 if fb['PriceDiff'].loc[ei] > 0 else 0 for ei in fb.index ]
fb['ma50'] = fb['Close'].rolling(50).mean()

#plot the moving average
plt.figure(figsize=(10, 8))
fb['ma50'].loc['2015-01-01':'2015-12-31'].plot(label='MA50')
fb['Close'].loc['2015-01-01':'2015-12-31'].plot(label='Close')
plt.legend()
plt.show()

#### Strategy
#import FB's stock data, add two columns - MA10 and MA50
#use dropna to remove any "Not a Number" data
fb = pd.DataFrame.from_csv('../data/facebook.csv')
fb['MA10'] = fb['Close'].rolling(10).mean()
fb['MA50'] = fb['Close'].rolling(50).mean()
fb = fb.dropna()
fb.head()

#Add a new column "Shares", if MA10>MA50, denote as 1 (long one share of stock), otherwise, denote as 0 (do nothing)

fb['Shares'] = [1 if fb.loc[ei, 'MA10']>fb.loc[ei, 'MA50'] else 0 for ei in fb.index]

#Add a new column "Profit" using List Comprehension, for any rows in fb, if Shares=1, the profit is calculated as the close price of
#tomorrow - the close price of today. Otherwise the profit is 0.

#Plot a graph to show the Profit/Loss

fb['Close1'] = fb['Close'].shift(-1)
fb['Profit'] = [fb.loc[ei, 'Close1'] - fb.loc[ei, 'Close'] if fb.loc[ei, 'Shares']==1 else 0 for ei in fb.index]
fb['Profit'].plot()
plt.axhline(y=0, color='red')

#Use .cumsum() to calculate the accumulated wealth over the period

fb['wealth'] = fb['Profit'].cumsum()
fb.tail()

#plot the wealth to show the growth of profit over the period

fb['wealth'].plot()
plt.title('Total money you win is {}'.format(fb.loc[fb.index[-2], 'wealth']))