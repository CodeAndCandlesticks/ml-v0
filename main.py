import pandas as pd
import numpy as np
import talib as ta
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import seaborn as sns

# The data is stored in the directory 'data_modules'
path = "candles/"

# Read the data
data = pd.read_csv(path + 'JPM_2017_2019.csv', index_col=0)
data.index = pd.to_datetime(data.index)

# Create a column 'future_returns' with the calculation of 
# percentage change
data['future_returns'] = data['close'].pct_change().shift(-1)

# Create the signal column
data['signal'] = np.where(data['future_returns'] > 0, 1, 0)

# Create a column 'pct_change' with the 15-minute prior 
# percentage change
data['pct_change'] = data['close'].pct_change()

# Create a column 'pct_change2' with the 30-minute prior 
# percentage change
data['pct_change2'] = data['close'].pct_change(2)

# Create a column 'pct_change5' with the 75-minute prior 
# percentage change
data['pct_change5'] = data['close'].pct_change(5)

## Creating the indicators using TA-lib

# Create a column by the name RSI, and assign the RSI values to it
data['rsi'] = ta.RSI(data['close'].values, timeperiod=int(6.5*4))

# Create a column by the name ADX, and assign the ADX values to it
data['adx'] = ta.ADX(data['high'].values, data['low'].values, 
                     data['open'].values, timeperiod=int(6.5*4))

# Create a column by the name sma, and assign SMA values to it
data['sma'] = data['close'].rolling(window=int(6.5*4)).mean()

# Create a column by the name corr, and assign the correlation 
# values to it
data['corr'] = data['close'].rolling(window=int(6.5*4))\
                .corr(data['sma'])

# 1-day and 2-day volatility
data['volatility'] = data.rolling(
    int(6.5*4), min_periods=int(6.5*4))['pct_change'].std()*100

data['volatility2'] = data.rolling(
    int(6.5*8), min_periods=int(6.5*8))['pct_change'].std()*100

# Dropping missing values, since we've calculated MAs
data.dropna(inplace=True)

## Storing the signal column in y and features in x:

# Target
y = data[['signal']].copy()

# Features
X = data[['open','high','low','close','pct_change', 
          'pct_change2', 'pct_change5', 'rsi', 'adx', 'sma', 
          'corr', 'volatility', 'volatility2']].copy()

#i=1
# Set number of rows in subplot
# nrows = int(X.shape[1]+1/2)
# for feature in X.columns:
#     plt.subplot(nrows, 2, i)
    
#     # Plot the feature
#     X[feature].plot(figsize=(8,3*X.shape[1]),
#                     color=np.random.rand(3,))
#     plt.ylabel(feature)
#     plt.title(feature)
#     i+=1
# plt.tight_layout()
# plt.show()

## Stationary checks
# As you have seen that most ML algorithm requires stationary features, we will drop the non-stationary features from X.

# You can use the adfuller method from the statsmodels library to perform this test in Python and compare the p-value.

# If the p-value is less than or equal to 0.05, you reject H0
# If the p-value is greater than 0.05, you fail to reject H0
def stationary(series):
    """Function to check if the series is stationary or not.
    """

    result = adfuller(series)
    if(result[1] < 0.05):
        return 'stationary'
    else:
        return 'not stationary'


# Check for stationarity
for col in X.columns:
    if stationary(data[col]) == 'not stationary':
        print('%s is not stationary. Dropping it.' % col)
        X.drop(columns=[col], axis=1, inplace=True)

# plt.figure(figsize=(8,5))
# sns.heatmap(X.corr(), annot=True, cmap='coolwarm')
# plt.show()

def get_pair_above_threshold(X, threshold):
    """Function to return the pairs with correlation above 
    threshold.
    """
    # Calculate the correlation matrix
    correl = X.corr()

    # Unstack the matrix
    correl = correl.abs().unstack()

    # Recurring & redundant pair
    pairs_to_drop = set()
    cols = X.corr().columns
    for i in range(0, X.corr().shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))

    # Drop the recurring & redundant pair
    correl = correl.drop(labels=pairs_to_drop) \
            .sort_values(ascending=False)

    return correl[correl > threshold].index


print(get_pair_above_threshold(X, 0.7))

# Drop the highly correlated column (as per above printed information)

X = X.drop(columns=['volatility2'], axis=1)

## Saving features 
X.to_csv ('data/JPM_2017_2019-features.csv')
##and Targets
y.to_csv ('data/JPM_2017_2019-target.csv')