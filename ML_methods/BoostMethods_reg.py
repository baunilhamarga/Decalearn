import numpy as np
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
import os
import time
import warnings

def load_data(file_path):
    tmp = np.load(file_path, allow_pickle=True)
    X_train = tmp['X_train']
    y_train = tmp['y_train']
    X_test = tmp['X_test']
    y_test = tmp['y_test']
    return X_train, y_train, X_test, y_test

def scale_data(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_train = np.expand_dims(y_train, axis=1)
    y_test = np.expand_dims(y_test, axis=1)

    scaler = StandardScaler()
    y_train = scaler.fit_transform(y_train)
    y_test = scaler.transform(y_test)
    
    return X_train, y_train, X_test, y_test

def run_XGBoost(file_path, X_train, y_train, X_test, y_test):
    X_train, y_train, X_test, y_test = scale_data(X_train, y_train, X_test, y_test)

    model = XGBRegressor()

    start_time = time.time()  # Record start time

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse_test = mean_squared_error(y_test, y_pred)

    end_time = time.time()  # Record end time
    elapsed_time = end_time - start_time

    file_name = os.path.basename(file_path)

    print(f'MSE on {file_name} (XGBoost): {mse_test}')
    print(f'Time elapsed for {file_name} (XGBoost): {elapsed_time}s\n')

def run_LightXGBoost(file_path, X_train, y_train, X_test, y_test):
    X_train, y_train, X_test, y_test = scale_data(X_train, y_train, X_test, y_test)

    start_time = time.time()  # Record start time

    # Suppress LightGBM warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lgb_train = LGBMRegressor(verbosity=-1).fit(X_train, y_train)
        y_pred = lgb_train.predict(X_test)
        mse_test = mean_squared_error(y_test, y_pred)

        end_time = time.time()  # Record end time
        elapsed_time = end_time - start_time

        file_name = os.path.basename(file_path)

        print(f'MSE on {file_name} (LightXGBoost): {mse_test}')
        print(f'Time elapsed for {file_name} (LightXGBoost): {elapsed_time}s\n')

def run_CatBoost(file_path, X_train, y_train, X_test, y_test):
    X_train, y_train, X_test, y_test = scale_data(X_train, y_train, X_test, y_test)

    start_time = time.time()  # Record start time

    # Suppress CatBoost output
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cat_model = CatBoostRegressor(verbose=0).fit(X_train, y_train)
        y_pred = cat_model.predict(X_test)
        mse_test = mean_squared_error(y_test, y_pred)

        end_time = time.time()  # Record end time
        elapsed_time = end_time - start_time

        file_name = os.path.basename(file_path)

        print(f'MSE on {file_name} (CatBoost): {mse_test}')
        print(f'Time elapsed for {file_name} (CatBoost): {elapsed_time}s\n')

if __name__ == '__main__':
    # Insert the file paths of regression datasets
    file_paths = [
        '/home/baunilha/Repositories/Decalearn/Datasets/Nasa Predictive Maintenance (RUL)/data_nasa.npz',
        '/home/baunilha/Repositories/Decalearn/Datasets/Sea Surface Height (SSH)/SSH_stationary.npz',
        '/home/baunilha/Repositories/Decalearn/Datasets/GeologyTasks/OilPrediction/cumulative_oil(indian)_stationary.npz',
    ]

    for file_path in file_paths:
        X_train, y_train, X_test, y_test = load_data(file_path)
        
        # Run XGBoost
        run_XGBoost(file_path, X_train, y_train, X_test, y_test)
        
        # Run LightXGBoost
        run_LightXGBoost(file_path, X_train, y_train, X_test, y_test)

        # Run CatBoost
        run_CatBoost(file_path, X_train, y_train, X_test, y_test)
