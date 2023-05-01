#import numpy and pandas package
import numpy as np
import pandas as pd
# roll two dice for multiple times
die = pd.DataFrame([1, 2, 3, 4, 5, 6])
sum_of_dice = die.sample(2, replace=True).sum().loc[0]
print('Sum of dice is', sum_of_dice)

# you may get different outcomes as we now mimic the result of rolling 2 dice, but the range must be limited between 2 and 12.
# It is your turn! let's replace the none with the code of rolling three dice, instead of two

np.random.seed(1)  # This is for checking answer, do NOT modify this line of code

#Modify the code, replace the None
sum_of_three_dice = die.sample(3, replace=True).sum().loc[0]
print('Sum of three dice is', sum_of_three_dice)
# The following code mimics the roll dice game for 50 times. And the results are all stored into "Result"
# Lets try and get the results of 50 sum of faces.

trial = 50
results = [die.sample(2, replace=True).sum().loc[0] for i in range(trial)]

#print the first 10 results
print(results[:10])
# This is the code for summarizing the results of sum of faces by frequency of outcome

freq = pd.DataFrame(results)[0].value_counts()
sort_freq = freq.sort_index()
print(sort_freq)
#plot the bar chart base on the result

sort_freq.plot(kind='bar', color='blue', figsize=(15, 8))
# Using relative frequency, we can rescale the frequency so that we can compare results from different number of trials
relative_freq = sort_freq/trial
relative_freq.plot(kind='bar', color='blue', figsize=(15, 8))
# Let us try to increase the number of trials to 10000, and see what will happen...
trial = 100
results = [die.sample(2, replace=True).sum().loc[0] for i in range(trial)]
freq = pd.DataFrame(results)[0].value_counts()
sort_freq = freq.sort_index()
relative_freq = sort_freq/trial
relative_freq.plot(kind='bar', color='blue', figsize=(15, 8))
# assume that we have fair dice, which means all faces will be shown with equal probability
# then we can say we know the 'Distribtuion' of the random variable - sum_of_dice

X_distri = pd.DataFrame(index=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
X_distri['Prob'] = [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]
X_distri['Prob'] = X_distri['Prob']/36
X_distri

mean = pd.Series(X_distri.index * X_distri['Prob']).sum()
var = pd.Series(((X_distri.index - mean)**2)*X_distri['Prob']).sum()
#Output the mean and variance of the distribution. Mean and variance can be used to describe a distribution
print(mean, var)
# if we calculate mean and variance of outcomes (with high enough number of trials, eg 20000)...
trial = 200
results = [die.sample(2, replace=True).sum().loc[0] for i in range(trial)]
#print the mean and variance of the 20000 trials
results = pd.Series(results)
print(results.mean(), results.var())