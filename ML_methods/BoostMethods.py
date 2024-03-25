import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
import os
import time
import warnings
from itertools import product

import sys
sys.path.append('..')
from Scripts import generate_coreset


random_state = 12227

def load_data(file_path, use_coreset=False, coreset_size=1024):
    tmp = np.load(file_path, allow_pickle=True)
    if use_coreset and len(tmp['X_train']) > coreset_size:
        X_train, y_train = generate_coreset.reconstruct_coreset(tmp['X_train'], tmp['y_train'], coreset_size)
    else:
        X_train = tmp['X_train']
        y_train = tmp['y_train']
    X_test = tmp['X_test']
    y_test = tmp['y_test']
    return X_train, y_train, X_test, y_test

def choose_parameters_XGBoost(X_train, y_train, timeout=True, timeout_seconds=60, verbose=False):
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=random_state)

    parameters_acc_list = []
    
    tested_params = {
                        'learning_rate': np.logspace(-7, 0, 8),
                        'max_depth': [1, 10],
                        'subsample': [0.2, 1],
                        'colsample_bytree': [0.2, 1],
                        'colsample_bylevel': [0.2, 1],
                        'min_child_weight': np.logspace(-16, 5, 22),
                        #'alpha': np.logspace(-16, 2, 19),
                        #'lambda': np.logspace(-16, 2, 19),
                        #'gamma': np.logspace(-16, 2, 19),
                        'n_estimators': [100, 4000],
                        'random_state': [random_state],
                    }

    total_params = np.prod([len(values) for values in tested_params.values()])

    start_time = time.time()  # Record start time

    # Generate all combinations of parameters
    for param_values in product(*tested_params.values()):
        parameters = dict(zip(tested_params.keys(), param_values))
        model = XGBClassifier(**parameters)
        model.fit(X_train, y_train)
        acc_val = accuracy_score(y_val, model.predict(X_val))
        parameters_acc_list.append((parameters, acc_val))
        # Print progress
        if verbose:
            print(f"tested {len(parameters_acc_list)}/{total_params} parameters", end='\r', flush=True)
        # Check timeout
        elapsed_time = time.time() - start_time
        if timeout and elapsed_time > timeout_seconds:
            # Find the tuple with the maximum accuracy
            best_parameters, best_accuracy = max(parameters_acc_list, key=lambda x: x[1])

            end_time = time.time()  # Record end time
            elapsed_time = end_time - start_time

            # Open a file for writing other outputs
            with open('logs/parameters.txt', 'a') as file:
                print(f"\nNumber of parameters processed: {len(parameters_acc_list)}/{total_params}", file=file)
                print("Timeout reached. Returning best parameters found so far.", file=file)
                print("Best Parameters:", best_parameters, file=file)
                print("Best Accuracy:", best_accuracy, file=file)
                print(f"Time elapsed to find Best Parameters (Timeout): {elapsed_time}s\n", file=file)
            return max(parameters_acc_list, key=lambda x: x[1])[0]

    # Find the tuple with the maximum accuracy
    best_parameters, best_accuracy = max(parameters_acc_list, key=lambda x: x[1])

    end_time = time.time()  # Record end time
    elapsed_time = end_time - start_time

    # Open a file for writing other outputs
    with open('logs/parameters.txt', 'a') as file:
        print("Best Parameters:", best_parameters, file=file)
        print("Best Accuracy:", best_accuracy, file=file)
        print(f"Time elapsed to find Best Parameters: {elapsed_time}s\n", file=file)

    return best_parameters

def choose_parameters_LightGBM(X_train, y_train, timeout=True, timeout_seconds=60, verbose=False):
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=random_state)

    parameters_acc_list = []
    
    tested_params = {
                        'num_leaves': range(5,50),
                        'max_depth': range(3, 20),
                        'learning_rate': np.logspace(-3, 0, 4),
                        'n_estimators': [50, 2000],
                        #'min_child_weight': np.logspace(-5, 4, 9),
                        #'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
                        #'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100],
                        'subsample': [0.2, 0.8],
                        'verbose': [-1],
                        'random_state': [random_state],
                    }
    
    total_params = np.prod([len(values) for values in tested_params.values()])

    start_time = time.time()  # Record start time

    for param_values in product(*tested_params.values()):
        parameters = dict(zip(tested_params.keys(), param_values))
        # Suppress LightGBM warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lgb_train = LGBMClassifier(**parameters).fit(X_train, y_train)
            y_pred = lgb_train.predict(X_val)
            acc_val = accuracy_score(y_val, y_pred)
            parameters_acc_list.append((parameters, acc_val))
        # Print progress
        if verbose:
            print(f"tested {len(parameters_acc_list)}/{total_params} parameters", end='\r', flush=True)
        # Check timeout
        elapsed_time = time.time() - start_time
        if timeout and elapsed_time > timeout_seconds:
            # Find the tuple with the maximum accuracy
            best_parameters, best_accuracy = max(parameters_acc_list, key=lambda x: x[1])

            end_time = time.time()  # Record end time
            elapsed_time = end_time - start_time

            # Open a file for writing other outputs
            with open('logs/parameters.txt', 'a') as file:
                print(f"\nNumber of parameters processed: {len(parameters_acc_list)}/{total_params}", file=file)
                print("Timeout reached. Returning best parameters found so far.", file=file)
                print("Best Parameters:", best_parameters, file=file)
                print("Best Accuracy:", best_accuracy, file=file)
                print(f"Time elapsed to find Best Parameters (Timeout): {elapsed_time}s\n", file=file)
            return max(parameters_acc_list, key=lambda x: x[1])[0]

    # Find the tuple with the maximum accuracy
    best_parameters, best_accuracy = max(parameters_acc_list, key=lambda x: x[1])

    end_time = time.time()  # Record end time
    elapsed_time = end_time - start_time

    # Open a file for writing other outputs
    with open('logs/parameters.txt', 'a') as file:
        print("Best Parameters:", best_parameters, file=file)
        print("Best Accuracy:", best_accuracy, file=file)
        print(f"Time elapsed to find Best Parameters: {elapsed_time}s\n", file=file)


    return best_parameters

def choose_parameters_CatBoost(X_train, y_train, timeout=True, timeout_seconds=60, verbose=False):
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=random_state)

    parameters_acc_list = []
    
    tested_params = {
                        'learning_rate': np.logspace(-5, 0, 6),
                        #'random_strength': [1, 20],
                        #'l2_leaf_reg': np.linspace(0, 10.0, 10),
                        #'bagging_temperature': np.linspace(0, 1.0, 10),
                        #'leaf_estimation_iterations': [1, 20],
                        #'iterations': [100, 4000],
                        'verbose': [0],
                        'random_state': [random_state],
                    }
    
    total_params = np.prod([len(values) for values in tested_params.values()])

    start_time = time.time()  # Record start time

    for param_values in product(*tested_params.values()):
        parameters = dict(zip(tested_params.keys(), param_values))
        # Suppress LightGBM warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cat_model = CatBoostClassifier(**parameters).fit(X_train, y_train)
            y_pred = cat_model.predict(X_val)
            acc_val = accuracy_score(y_val, y_pred)
            parameters_acc_list.append((parameters, acc_val))
        # Print progress
        if verbose:
            print(f"tested {len(parameters_acc_list)}/{total_params} parameters", end='\r', flush=True)
        # Check timeout
        elapsed_time = time.time() - start_time
        if timeout and elapsed_time > timeout_seconds:
            # Find the tuple with the maximum accuracy
            best_parameters, best_accuracy = max(parameters_acc_list, key=lambda x: x[1])

            end_time = time.time()  # Record end time
            elapsed_time = end_time - start_time

            # Open a file for writing other outputs
            with open('logs/parameters.txt', 'a') as file:
                print(f"\nNumber of parameters processed: {len(parameters_acc_list)}/{total_params}", file=file)
                print("Timeout reached. Returning best parameters found so far.", file=file)
                print("Best Parameters:", best_parameters, file=file)
                print("Best Accuracy:", best_accuracy, file=file)
                print(f"Time elapsed to find Best Parameters (Timeout): {elapsed_time}s\n", file=file)
            return max(parameters_acc_list, key=lambda x: x[1])[0]

    # Find the tuple with the maximum accuracy
    best_parameters, best_accuracy = max(parameters_acc_list, key=lambda x: x[1])

    end_time = time.time()  # Record end time
    elapsed_time = end_time - start_time

    # Open a file for writing other outputs
    with open('logs/parameters.txt', 'a') as file:
        print("Best Parameters:", best_parameters, file=file)
        print("Best Accuracy:", best_accuracy, file=file)
        print(f"Time elapsed to find Best Parameters: {elapsed_time}s\n", file=file)


    return best_parameters

def run_XGBoost(file_path, X_train, y_train, X_test, y_test, default=True, timeout=True, timeout_seconds=60, verbose=False):
    file_name = os.path.basename(file_path)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if default:
        best_parameters = {'random_state': random_state}
        #best_parameters = {'learning_rate': 1.0, 'max_depth': 10, 'subsample': 1, 'colsample_bytree': 1, 'colsample_bylevel': 1, 'min_child_weight': 0.001, 'n_estimators': 4000, 'random_state': 12227} #facies
    else:
        with open('logs/parameters.txt', 'a') as file:
            print(f"Finding best parameters for XGBoost on {file_name}...", file=file)
        best_parameters = choose_parameters_XGBoost(X_train, y_train, timeout=timeout, timeout_seconds=timeout_seconds, verbose=verbose)

    model = XGBClassifier(**best_parameters)

    start_time = time.time()  # Record start time

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc_test = accuracy_score(y_test, y_pred)

    end_time = time.time()  # Record end time
    elapsed_time = end_time - start_time

    print(f'Accuracy on {file_name} (XGBoost): {acc_test}')
    print(f'Time elapsed for {file_name} (XGBoost): {elapsed_time}s\n\n')

def run_LightGBM(file_path, X_train, y_train, X_test, y_test, default=True, timeout=True, timeout_seconds=60, verbose=False):
    file_name = os.path.basename(file_path)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Suppress LightGBM warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if default:
            best_parameters = {'verbosity': -1, 'random_state': random_state}
            #best_parameters = {'num_leaves': 49, 'max_depth': 7, 'learning_rate': 0.1, 'n_estimators': 2000, 'min_child_weight': 1e-05, 'subsample': 0.2, 'verbose': -1, 'random_state': 12227}
        else:
            with open('logs/parameters.txt', 'a') as file:
                print(f"Finding best parameters for LightGBM on {file_name}...", file=file)
            best_parameters = choose_parameters_LightGBM(X_train, y_train, timeout=timeout, timeout_seconds=timeout_seconds, verbose=verbose)

        start_time = time.time()  # Record start time
        
        lgb_train = LGBMClassifier(**best_parameters).fit(X_train, y_train)
        y_pred = lgb_train.predict(X_test)
        acc_test = accuracy_score(y_test, y_pred)

        end_time = time.time()  # Record end time
        elapsed_time = end_time - start_time

        print(f'Accuracy on {file_name} (LightGBM): {acc_test}')
        print(f'Time elapsed for {file_name} (LightGBM): {elapsed_time}s\n\n')

def run_CatBoost(file_path, X_train, y_train, X_test, y_test, default=True, timeout=True, timeout_seconds=60, verbose=False):
    file_name = os.path.basename(file_path)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Suppress CatBoost output
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if default:
            best_parameters = {'verbose': 0, 'random_state': random_state}
            #best_parameters = {'allow_writing_files': False, 'verbose': False, 'iterations': 6407, 'use_best_model': False, 'early_stopping_rounds': None, 'border_count': 25425, 'colsample_bylevel': 0.06267846249024994, 'l2_leaf_reg': 7.980471584955271, 'learning_rate': 0.23176336497082237, 'max_depth': 7, 'max_leaves': 31, 'min_data_in_leaf': 137337.0, 'random_state': random_state}  # Results from MAHD AutoFedot 37h
        else:
            with open('logs/parameters.txt', 'a') as file:
                print(f"Finding best parameters for CatBoost on {file_name}...", file=file)
            best_parameters = choose_parameters_CatBoost(X_train, y_train, timeout=timeout, timeout_seconds=timeout_seconds, verbose=verbose)

        start_time = time.time()  # Record start time
        
        cat_model = CatBoostClassifier(**best_parameters).fit(X_train, y_train)
        y_pred = cat_model.predict(X_test)
        acc_test = accuracy_score(y_test, y_pred)

        end_time = time.time()  # Record end time
        elapsed_time = end_time - start_time

        print(f'Accuracy on {file_name} (CatBoost): {acc_test}')
        print(f'Time elapsed for {file_name} (CatBoost): {elapsed_time}s\n\n')

if __name__ == '__main__':
    # Insert the file paths of classification datasets
    file_paths = [
        '../Datasets/Multimodal Human Action/data/UTD-MHAD2_1s.npz',
        #'../Datasets/GeologyTasks/FaciesClassification/FaciesClassificationYananGasField.npz',
        #'../Datasets/Lucas/osha_train_test.npz',
        #'../Datasets/AI_text/AI_text.npz',
        #'../Datasets/CSI/dataset_csi.npz',
    ]

    for file_path in file_paths:
        X_train, y_train, X_test, y_test = load_data(file_path, use_coreset=False)

        # Run XGBoost
        run_XGBoost(file_path, X_train, y_train, X_test, y_test, default=True, verbose=True, timeout=False, timeout_seconds=10)
    
        # Run LightGBM
        run_LightGBM(file_path, X_train, y_train, X_test, y_test, default=True, verbose=True, timeout=False, timeout_seconds=10)

        # Run CatBoost
        run_CatBoost(file_path, X_train, y_train, X_test, y_test, default=True, verbose=True, timeout=False, timeout_seconds=10)
