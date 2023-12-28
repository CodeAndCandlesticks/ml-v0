import pandas as pd
import numpy as np
import talib as ta
#from statsmodels.tsa.stattools import adfuller

# The data is stored in the directory 'data_modules'
path = "candles/"

# Read the data
data = pd.read_csv(path + 'JPM_2017_2019.csv', index_col=0)
data.index = pd.to_datetime(data.index)

print(data.head())