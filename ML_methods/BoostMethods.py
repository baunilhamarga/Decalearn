import numpy as np
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
import os
import time
import warnings

def load_data(file_path):
    tmp = np.load(file_path)
    X_train = tmp['X_train']
    y_train = tmp['y_train']
    X_test = tmp['X_test']
    y_test = tmp['y_test']
    return X_train, y_train, X_test, y_test

def run_XGBoost(file_path, X_train, y_train, X_test, y_test):
    n_classes = len(np.unique(y_train))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = XGBClassifier()

    start_time = time.time()  # Record start time

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc_test = accuracy_score(y_test, y_pred)

    end_time = time.time()  # Record end time

    elapsed_time = end_time - start_time

    file_name = os.path.basename(file_path)

    print(f'Accuracy on {file_name} (XGBoost): {acc_test}')
    print(f'Time elapsed for {file_name} (XGBoost): {elapsed_time}s\n')

def run_LightXGBoost(file_path, X_train, y_train, X_test, y_test):
    n_classes = len(np.unique(y_train))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    start_time = time.time()  # Record start time

    # Suppress LightGBM warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lgb_train = LGBMClassifier(verbosity=-1).fit(X_train, y_train)
        y_pred = lgb_train.predict(X_test)
        acc_test = accuracy_score(y_test, y_pred)

        end_time = time.time()  # Record end time
        elapsed_time = end_time - start_time

        file_name = os.path.basename(file_path)

        print(f'Accuracy on {file_name} (LightXGBoost): {acc_test}')
        print(f'Time elapsed for {file_name} (LightXGBoost): {elapsed_time}s\n')

def run_CatBoost(file_path, X_train, y_train, X_test, y_test):
    n_classes = len(np.unique(y_train))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    start_time = time.time()  # Record start time

    # Suppress CatBoost output
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cat_model = CatBoostClassifier(verbose=0).fit(X_train, y_train)
        y_pred = cat_model.predict(X_test)
        acc_test = accuracy_score(y_test, y_pred)

        end_time = time.time()  # Record end time
        elapsed_time = end_time - start_time

        file_name = os.path.basename(file_path)

        print(f'Accuracy on {file_name} (CatBoost): {acc_test}')
        print(f'Time elapsed for {file_name} (CatBoost): {elapsed_time}s\n')

if __name__ == '__main__':
    # Insert the file paths of classification datasets
    file_paths = [
        '/home/baunilha/Repositories/Decalearn/Datasets/Multimodal Human Action/data/UTD-MHAD2_1s.npz',
        '/home/baunilha/Repositories/Decalearn/Datasets/GeologyTasks/FaciesClassification/FaciesClassificationYananGasField.npz',
    ]

    for file_path in file_paths:
        X_train, y_train, X_test, y_test = load_data(file_path)
        
        # Run XGBoost
        run_XGBoost(file_path, X_train, y_train, X_test, y_test)
        
        # Run LightXGBoost
        run_LightXGBoost(file_path, X_train, y_train, X_test, y_test)

        # Run CatBoost
        run_CatBoost(file_path, X_train, y_train, X_test, y_test)
