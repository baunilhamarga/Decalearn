# https://www.kaggle.com/code/wassimderbel/nasa-predictive-maintenance-rul/notebook

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import sklearn
import os
from sklearn.model_selection import train_test_split
import random
import warnings
from sklearn.preprocessing import MinMaxScaler
np.random.seed(12227)
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def preprocess_and_save_data(file_name = 'data_nasa', scale = False):
    """
    Imports NASA RUL data, preprocesses it by removing unnecessary features and stores it in an .npz file.

    Parameters:
    - file_name: (srt) Name of the generated file
    - scale: (bool) If set True, the data arrays will be scaled
    """
    # Define feature names for internal organization (labels will be dropped later)
    index_names = ['unit_number', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i+1) for i in range(0, 21)]
    col_names = index_names + setting_names + sensor_names

    # Import train and validation data
    # FD001 subset corresponds to HPC engine failure.
    # Note: train data will be split into train/test sets in the modeling part.
    df_train = pd.read_csv('Cmaps/train_FD001.txt', sep='\s+', header=None, index_col=False, names=col_names)
    df_valid = pd.read_csv('Cmaps/test_FD001.txt', sep='\s+', header=None, index_col=False, names=col_names)
    y_val = pd.read_csv('Cmaps/RUL_FD001.txt', sep='\s+', header=None, index_col=False, names=['RUL'])

    train = df_train.copy()
    valid = df_valid.copy()

    # Add RUL column to the data (target)
    def add_RUL_column(df):
        train_grouped_by_unit = df.groupby(by='unit_number')
        max_time_cycles = train_grouped_by_unit['time_cycles'].max()
        merged = df.merge(max_time_cycles.to_frame(name='max_time_cycle'), left_on='unit_number', right_index=True)
        merged["RUL"] = merged["max_time_cycle"] - merged['time_cycles']
        merged = merged.drop("max_time_cycle", axis=1)
        return merged

    train = add_RUL_column(train)

    # Drop unnecessary features (constant sensors, labels, and settings)
    dropped_sensors = ['s_{}'.format(i) for i in [1, 5, 6, 10, 16, 18, 19]]
    dropped_labels = index_names + setting_names
    X_train = train.drop(columns=dropped_sensors + dropped_labels).copy()

    # Keep only the last occurrence of each unit to match the length of y_valid
    X_val = valid.groupby('unit_number').last().reset_index().drop(columns=dropped_labels + dropped_sensors)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_train, X_train['RUL'], test_size=0.3, random_state=12227)

    # Drop the target feature
    X_train.drop(columns=['RUL'], inplace=True)
    X_test.drop(columns=['RUL'], inplace=True)

    if scale:
        X_train, X_test, X_val = scale_data(X_train, X_test, X_val)
        file_name = 'data_nasa_scaled'

    # Save the compressed data
    np.savez_compressed(file_name, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, X_val=X_val, y_val=y_val)


def scale_data(X_train, X_test, X_val):
    """
    Scale the input data using Min-Max scaling.

    Parameters:
    - X_train: Training data array.
    - X_test: Testing data array.
    - X_val: Validation data array.

    Returns:
    - X_train_s: Scaled training data.
    - X_test_s: Scaled testing data.
    - X_val_s: Scaled validation data.
    """
    # Create a MinMaxScaler instance
    scaler = MinMaxScaler()

    # Fit and transform the training data
    X_train_s = scaler.fit_transform(X_train)

    # Transform the testing data (use the same scaling parameters as the training data)
    X_test_s = scaler.transform(X_test)

    # Transform the validation data (use the same scaling parameters as the training data)
    X_val_s = scaler.transform(X_val)

    return X_train_s, X_test_s, X_val_s


if __name__ == '__main__':
    preprocess_and_save_data()
    data_nasa = np.load('data_nasa.npz')
    print(data_nasa['X_train'])
    preprocess_and_save_data(scale=True)
    data_nasa = np.load('data_nasa_scaled.npz')
    print(data_nasa['X_train'])