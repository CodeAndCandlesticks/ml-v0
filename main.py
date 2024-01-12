import pandas as pd
import numpy as np
import talib as ta
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import seaborn as sns
import argparse
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pickle


def load_data(file_path):
    data = pd.read_csv(file_path, index_col=0) # consider adding utc=True for timeseries with timezone information
    data.index = pd.to_datetime(data.index)

    return data

def feature_engineering(data, base_filename):
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
    
    i=1

    # Set number of rows in subplot
    nrows = int(X.shape[1]+1/2)
    for feature in X.columns:
        plt.subplot(nrows, 2, i)
        
        # Plot the feature
        X[feature].plot(figsize=(8,3*X.shape[1]),
                        color=np.random.rand(3,))
        plt.ylabel(feature)
        plt.title(feature)
        i+=1
    plt.tight_layout()
    #plt.show() #not supported in WSL
    plt.savefig(f'models/{base_filename}-features.png')
    
    X = remove_non_stationary_columns(X)
    
    X = remove_highly_correlated(X, 0.7, base_filename)
    


    return y, X

def remove_highly_correlated(df, threshold, base_filename):
    """Removes columns with correlation higher than the specified threshold.
    
    Args:
    df (DataFrame): The input DataFrame with features.
    threshold (float): The threshold for correlation.

    Returns:
    DataFrame: The DataFrame with highly correlated columns removed.
    """
    plt.figure(figsize=(8,5))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    #plt.show() # not supported by WSL
    plt.savefig(f'models/{base_filename}-correlation_matrix.png')

    # Calculate the correlation matrix and get the absolute value
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find columns with correlation greater than the threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    # Drop columns
    df_dropped = df.drop(columns=to_drop, axis=1)

    return df_dropped

def remove_non_stationary_columns(df, significance_level=0.05):
    """Removes non-stationary columns from the DataFrame.

    Args:
    df (DataFrame): The input DataFrame.
    significance_level (float): The significance level for the ADF test.

    Returns:
    DataFrame: A DataFrame with non-stationary columns removed.
    """
    non_stationary_columns = []

    for col in df.columns:
        result = adfuller(df[col])
        if result[1] >= significance_level:
            non_stationary_columns.append(col)
            print(f'{col} is not stationary. Dropping it.')

    # Drop non-stationary columns
    df.drop(columns=non_stationary_columns, inplace=True)
    return df

def save_data(dataframe, file_path, boolean=True):
    """Saves the given DataFrame to a CSV file.

    Args:
    dataframe (pd.DataFrame): The DataFrame to be saved.
    file_path (str): The file path where the DataFrame should be saved.
    """
    dataframe.to_csv(file_path, index=boolean)
    print(f'Data saved to {file_path}')

def split_data (feature_file, target_file, train_size, base_filename):
    X = pd.read_csv(feature_file, index_col=0, parse_dates=True)
    y = pd.read_csv(target_file, index_col=0, parse_dates=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, shuffle=False) # No shuffling because the indices in time series data are timestamps that occur one after the other (in sequence).
    
    # Plot the data
    plt.figure(figsize=(8, 5))

    plt.plot(X_train['pct_change'], linestyle='None',
            marker='.', markersize=3.0, label='X_train data', 
            color='blue')
    plt.plot(X_test['pct_change'], linestyle='None',
            marker='.', markersize=3.0, label='X_test data', 
            color='green')

    # Set the title and axis label
    plt.title("Visualising Train and Test Datasets (pct_change Column)", 
            fontsize=14)
    plt.xlabel('Years', fontsize=12)
    plt.ylabel('% change (%)', fontsize=12)

    # Display the plot
    plt.legend()
    #plt.show() # not supported in WSL
    plt.savefig(f'models/{base_filename}-test-and-train-data.png')

    return X_train, X_test, y_train, y_test

def create_and_train (feature_training_file, target_training_file):
    X_train = pd.read_csv (feature_training_file, index_col=0, parse_dates=True)
    y_train = pd.read_csv (target_training_file, index_col=0, parse_dates=True)
    model = RandomForestClassifier (n_estimators = 3, max_features=3, max_depth=2, random_state=4)
    model.fit (X_train, y_train['signal'])

    return model

def save_model(model, model_name):
    """Saves the given machine learning model into the models/ directory.

    Args:
    model (any sklearn model or similar): The trained machine learning model to be saved.
    model_name (str): The name for the saved model file.
    """
    # Ensure the models/ directory exists
    if not os.path.exists('models'):
        os.makedirs('models')

    # Define the path for saving the model
    model_path = os.path.join('models', f'{model_name}.pkl')

    # Save the model
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
    print(f'Model saved to {model_path}')

def evaluation_metrics (predicted_values, expected_values, base_filename):
    y_predicted = pd.read_csv(predicted_values, index_col=0, parse_dates=True)['signal']

    y_expected = pd.read_csv(expected_values, index_col=0, parse_dates=True)['signal']
    # Define the accuracy data
    accuracy_data = (y_predicted == y_expected)

    # Accuracy percentage
    accuracy_percentage = round(100 * accuracy_data.sum()/len(accuracy_data), 2)
    cm = confusion_matrix(y_expected.values, y_predicted.values)

    # Calculate percentages
    win = cm[1,1]
    loss = cm[0,1]
    loss_prevented = cm[0,0]
    opportunity_loss = cm[1,0]
    win_rate = round(100 * win / cm[1].sum(), 2)  # True Positives / Total Actual Positives
    loss_rate = round(100 * loss / cm[0].sum(), 2)  # False Positives / Total Actual Negatives

    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    #plt.show() # not supported by WSL
    plt.savefig(f'models/{base_filename}-cm.png')
    print(f'Wins: {win}')
    print(f'Losses: {loss}')
    print(f'Losses Prevented: {loss_prevented}')
    print(f'Opportunities Lost: {opportunity_loss}')
    print(f'Win_rate: {win_rate}')
    print(f'Loss rate: {loss_rate}')

    print (classification_report (y_expected.values, y_predicted.values))
    
def main():
    parser = argparse.ArgumentParser(description='Process some candle file.')
    parser.add_argument('file_path', type=str, help='Path to the candles file')
    args = parser.parse_args()

    # Generating file names based on input
    base_filename = os.path.basename(args.file_path).replace('.csv', '')
    feature_file = os.path.join('data', f'{base_filename}-features.csv')
    target_file = os.path.join('data', f'{base_filename}-target.csv')
    feature_training_file = os.path.join ('data', f'{base_filename}-features-train.csv')
    feature_testing_file = os.path.join ('data', f'{base_filename}-features-test.csv')
    target_training_file = os.path.join ('data', f'{base_filename}-target-train.csv')
    target_testing_file = os.path.join ('data', f'{base_filename}-target-test.csv')
    target_predicted_file = os.path.join ('data', f'{base_filename}-target-predict.csv')

    # Check if files already exist
    if not os.path.exists(feature_file) or not os.path.exists(target_file):
        print ("Data load and Feature engineering")
        data = load_data(args.file_path)
        y, X = feature_engineering(data, base_filename)
        save_data(X, feature_file)
        save_data(y, target_file)
    else:
        print("Feature and target files already exist, moving on to split the data")
    
    if not os.path.exists (feature_training_file) or not os.path.exists(feature_testing_file) or not os.path.exists (target_training_file) or not os.path.exists (target_testing_file):
        print ("Splitting the data")
        X_train, X_test, y_train, y_test = split_data (feature_file=feature_file, target_file=target_file, train_size=0.8, base_filename=base_filename)
        save_data (X_train,feature_training_file)
        save_data (X_test, feature_testing_file)
        save_data (y_train, target_training_file)
        save_data (y_test,target_testing_file)
    else:
        print ("Training and Test data already split, moving to training")

    if not os.path.exists (target_predicted_file):
        my_first_model = create_and_train (feature_training_file=feature_training_file, target_training_file=target_training_file)
        X_test = pd.read_csv (feature_testing_file, index_col=0, parse_dates=True)
        y_predicted = my_first_model.predict (X_test)
        y_predicted_df = pd.DataFrame(y_predicted, index=X_test.index)
        y_predicted_df.columns = ['signal']
        save_data (y_predicted_df, target_predicted_file, True)
        save_model(my_first_model, base_filename) 
    else:
        print ("Model trained and prediction made, model available on models folder")
    
    evaluation_metrics (target_predicted_file, target_testing_file, base_filename)



if __name__ == "__main__":
    main()
