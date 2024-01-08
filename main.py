import pandas as pd
import numpy as np
import talib as ta
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import seaborn as sns
import argparse
import os
from sklearn.model_selection import train_test_split


def load_data(file_path):
    data = pd.read_csv(file_path, index_col=0)
    data.index = pd.to_datetime(data.index)
    # ... other data preprocessing steps ...
    return data

def feature_engineering(data):
    """
    Process the input data to extract and engineer features and target.

    Args:
    data (DataFrame): The input data.

    Returns:
    tuple: A tuple containing the target DataFrame (y) and features DataFrame (X).
    """
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

    ## Storing the signal column in y and features in X:    
    # Target
    y = data[['signal']].copy()

    # Features
    X = data[['open','high','low','close','pct_change', 
            'pct_change2', 'pct_change5', 'rsi', 'adx', 'sma', 
            'corr', 'volatility', 'volatility2']].copy()
    
    X = remove_non_stationary_columns(X)

    X = remove_highly_correlated(X, 0.7)

    return y, X

def remove_highly_correlated(df, threshold):
    """Removes columns with correlation higher than the specified threshold.
    
    Args:
    df (DataFrame): The input DataFrame with features.
    threshold (float): The threshold for correlation.

    Returns:
    DataFrame: The DataFrame with highly correlated columns removed.
    """
    # Calculate the correlation matrix and get the absolute value
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find columns with correlation greater than the threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    # Drop columns
    df_dropped = df.drop(columns=to_drop, axis=1)

    return df_dropped

def save_data(data, file_path):
    data.to_csv(file_path)

def remove_non_stationary_columns(df, significance_level=0.05):
    """Removes non-stationary columns from the DataFrame.

    Args:
    df (DataFrame): The input DataFrame.
    significance_level (float): The significance level for the ADF test.

    Returns:
    DataFrame: A DataFrame with non-stationary columns removed.
    """
    stationary_columns = []

    for col in df.columns:
        result = adfuller(df[col])
        if result[1] < significance_level:
            stationary_columns.append(col)
        else:
            print(f'{col} is not stationary. Dropping it.')

    return df[stationary_columns]

def split_data (feature_file, target_file, train_size):
    X = pd.read_csv(feature_file, index_col=0, parse_dates=True)
    y = pd.read_csv(target_file, index_col=0, parse_dates=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, shuffle=False) # No shuffling because the indices in time series data are timestamps that occur one after the other (in sequence).
    
    # # Plot the data
    # plt.figure(figsize=(8, 5))

    # plt.plot(X_train['pct_change'], linestyle='None',
    #         marker='.', markersize=3.0, label='X_train data', 
    #         color='blue')
    # plt.plot(X_test['pct_change'], linestyle='None',
    #         marker='.', markersize=3.0, label='X_test data', 
    #         color='green')

    # # Set the title and axis label
    # plt.title("Visualising Train and Test Datasets (pct_change Column)", 
    #         fontsize=14)
    # plt.xlabel('Years', fontsize=12)
    # plt.ylabel('% change (%)', fontsize=12)

    # # Display the plot
    # plt.legend()
    # plt.show()

    return X_train, X_test, y_train, y_test


def main():
    parser = argparse.ArgumentParser(description='Process some candle file.')
    parser.add_argument('file_path', type=str, help='Path to the candles file')
    args = parser.parse_args()

    # Generating file names based on input
    base_filename = os.path.basename(args.file_path).replace('.csv', '')
    feature_file = os.path.join('data', f'{base_filename}-features.csv')
    target_file = os.path.join('data', f'{base_filename}-target.csv')
    
    # Check if files already exist
    if not os.path.exists(feature_file) or not os.path.exists(target_file):
        print ("Data load and Feature engineering")
        data = load_data(args.file_path)
        y, X = feature_engineering(data)
        save_data(X, feature_file)
        save_data(y, target_file)
    else:
        print("Feature and target files already exist, moving on to split the data")

    feature_training_file = os.path.join ('data', f'{base_filename}-features-train.csv')
    feature_testing_file = os.path.join ('data', f'{base_filename}-features-test.csv')
    target_training_file = os.path.join ('data', f'{base_filename}-target-train.csv')
    target_testing_file = os.path.join ('data', f'{base_filename}-target-test.csv')
    
    if not os.path.exists (feature_training_file) or not os.path.exists(feature_testing_file) or not os.path.exists (target_training_file) or not os.path.exists (target_testing_file):
        print ("Splitting the data")
        X_train, X_test, y_train, y_test = split_data (feature_file=feature_file, target_file=target_file, train_size=0.8)
        save_data (X_train,feature_training_file)
        save_data (X_test, feature_testing_file)
        save_data (y_train, target_training_file)
        save_data (y_test,target_testing_file)
    else:
        print ("Training and Test data already split, moving to training")




if __name__ == "__main__":
    main()
