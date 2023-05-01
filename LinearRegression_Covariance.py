import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


housing = pd.DataFrame.from_csv('../data/housing.csv', index_col=0)
housing.head()

# LSTAT	INDUS	NOX	RM	MEDV
# 0	4.98	2.31	0.538	6.575	24.0
# 1	9.14	7.07	0.469	6.421	21.6
# 2	4.03	7.07	0.469	7.185	34.7
# 3	2.94	2.18	0.458	6.998	33.4
# 4	5.33	2.18	0.458	7.147	36.2

# Use covariance to calculate the association

housing.cov()
# 	LSTAT	INDUS	NOX	RM	MEDV
# LSTAT	50.994760	29.580270	0.488946	-3.079741	-48.447538
# INDUS	29.580270	47.064442	0.607074	-1.887957	-30.520823
# NOX	0.488946	0.607074	0.013428	-0.024603	-0.455412
# RM	-3.079741	-1.887957	-0.024603	0.493671	4.493446
# MEDV	-48.447538	-30.520823	-0.455412	4.493446	84.586724

# Use correlation to calculate the association is more appropriate in this case
housing.corr()

# 	LSTAT	INDUS	NOX	RM	MEDV
# LSTAT	1.000000	0.603800	0.590879	-0.613808	-0.737663
# INDUS	0.603800	1.000000	0.763651	-0.391676	-0.483725
# NOX	0.590879	0.763651	1.000000	-0.302188	-0.427321
# RM	-0.613808	-0.391676	-0.302188	1.000000	0.695360
# MEDV	-0.737663	-0.483725	-0.427321	0.695360	1.000000

# scatter matrix plot

#from pandas.tools.plotting import scatter_matrix
#sm = scatter_matrix(housing, figsize=(10, 10))


# This time we take a closer look at MEDV vs LSTAT。 What is the association between MEDV and LSTAT you observed?
housing.plot(kind='scatter', x='LSTAT', y='MEDV', figsize=(10, 10))

# Simple linear regression
# yi=β0+β1∗xi+ϵi

# We shall base on the association between LSTAT and MEDV and create a simple linear regression model.
# Let's use python in estimating the values of B0 and B1 (intercept and slope)
# lets try to guess what are the real values of intercept and slope
# we call our guess b0, b1...
# Try to assign the value of b0, b1 to get a straight line that can describe our data
b0 = 0.1
b1 = 1
housing['GuessResponse'] = b0 + b1*housing['RM']

# Also want to know the error of guess...
# This show how far is our guess response from the true response
housing['observederror'] = housing['MEDV'] - housing['GuessResponse']

# plot your estimated line together with the points
plt.figure(figsize=(10, 10))
plt.title('Sum of sqaured error is {}'.format((((housing['observederror'])**2)).sum()))
plt.scatter(housing['RM'], housing['MEDV'], color='g', label='Observed')
plt.plot(housing['RM'], housing['GuessResponse'], color='red', label='GuessResponse')
plt.legend()
plt.xlim(housing['RM'].min()-2, housing['RM'].max()+2)
plt.ylim(housing['MEDV'].min()-2, housing['MEDV'].max()+2)
plt.show()
#Least square estimates
# Input the formula (refer to the lecture video 4.3)
formula = None
model = smf.ols(formula=formula, data=housing).fit()

# Here are estimated intercept and slope by least square estimation
# Attribute 'params' returns a list of estimated parameters form model
b0_ols = model.params[0]
b1_ols = model.params[1]

housing['BestResponse'] = b0_ols + b1_ols*housing['RM']

# Also want to know the error of of guess...
housing['error'] = housing['MEDV'] - housing['BestResponse']


# plot your estimated line together with the points
plt.figure(figsize=(10, 10))
# See if the error drops after you use least square method
plt.title('Sum of sqaured error is {}'.format((((housing['error'])**2)).sum()))
plt.scatter(housing['RM'], housing['MEDV'], color='g', label='Observed')
plt.plot(housing['RM'], housing['GuessResponse'], color='red', label='GuessResponse')
plt.plot(housing['RM'], housing['BestResponse'], color='yellow', label='BestResponse')
plt.legend()
plt.xlim(housing['RM'].min()-2, housing['RM'].max()+2)
plt.ylim(housing['MEDV'].min()-2, housing['MEDV'].max()+2)
plt.show()

#Refer to the P-value of RM, Confidence Interval and R-square to evaluate the performance.
model.summary()

# OLS Regression Results
# Dep. Variable:	MEDV	R-squared:	0.484
# Model:	OLS	Adj. R-squared:	0.483
# Method:	Least Squares	F-statistic:	471.8
# Date:	Sun, 30 Jun 2019	Prob (F-statistic):	2.49e-74
# Time:	12:21:18	Log-Likelihood:	-1673.1
# No. Observations:	506	AIC:	3350.
# Df Residuals:	504	BIC:	3359.
# Df Model:	1
# Covariance Type:	nonrobust
# coef	std err	t	P>|t|	[0.025	0.975]
# Intercept	-34.6706	2.650	-13.084	0.000	-39.877	-29.465
# RM	9.1021	0.419	21.722	0.000	8.279	9.925
# Omnibus:	102.585	Durbin-Watson:	0.684
# Prob(Omnibus):	0.000	Jarque-Bera (JB):	612.449
# Skew:	0.726	Prob(JB):	1.02e-133
# Kurtosis:	8.190	Cond. No.	58.4

# Diagnostic of models
model = smf.ols(formula='MEDV~LSTAT', data=housing).fit()

# Here are estimated intercept and slope by least square estimation
b0_ols = model.params[0]
b1_ols = model.params[1]

housing['BestResponse'] = b0_ols + b1_ols*housing['LSTAT']

# Assumptions behind linear regression model
# 1. Linearity
# 2. independence
# 3. Normality
# 4. Equal Variance

# Linerarity
# you can check the scatter plot to have a fast check
housing.plot(kind='scatter', x='LSTAT', y='MEDV', figsize=(10, 10), color='g')

# Independence
# Get all errors (residuals)
housing['error'] = housing['MEDV'] - housing['BestResponse']
# Method 1: Residual vs order plot
# error vs order plot (Residual vs order) as a fast check
plt.figure(figsize=(15, 8))
plt.title('Residual vs order')
plt.plot(housing.index, housing['error'], color='purple')
plt.axhline(y=0, color='red')
plt.show()
# Method 2: Durbin Watson Test
# Check the Durbin Watson Statistic
# Rule of thumb: test statistic value in the range of 1.5 to 2.5 are relatively normal
model.summary()

# OLS Regression Results
# Dep. Variable:	MEDV	R-squared:	0.544
# Model:	OLS	Adj. R-squared:	0.543
# Method:	Least Squares	F-statistic:	601.6
# Date:	Sun, 15 Sep 2019	Prob (F-statistic):	5.08e-88
# Time:	07:26:39	Log-Likelihood:	-1641.5
# No. Observations:	506	AIC:	3287.
# Df Residuals:	504	BIC:	3295.
# Df Model:	1
# Covariance Type:	nonrobust
# coef	std err	t	P>|t|	[0.025	0.975]
# Intercept	34.5538	0.563	61.415	0.000	33.448	35.659
# LSTAT	-0.9500	0.039	-24.528	0.000	-1.026	-0.874
# Omnibus:	137.043	Durbin-Watson:	0.892
# Prob(Omnibus):	0.000	Jarque-Bera (JB):	291.373
# Skew:	1.453	Prob(JB):	5.36e-64
# Kurtosis:	5.319	Cond. No.	29.7

# Normality
import scipy.stats as stats
z = (housing['error'] - housing['error'].mean())/housing['error'].std(ddof=1)

stats.probplot(z, dist='norm', plot=plt)
plt.title('Normal Q-Q plot')
plt.show()

# Equal variance
# Residual vs predictor plot
housing.plot(kind='scatter', x='LSTAT', y='error', figsize=(15, 8), color='green')
plt.title('Residual vs predictor')
plt.axhline(y=0, color='red')
plt.show()

