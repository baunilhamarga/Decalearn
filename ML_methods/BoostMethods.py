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

random_state = 12227

def load_data(file_path):
    tmp = np.load(file_path)
    X_train = tmp['X_train']
    y_train = tmp['y_train']
    X_test = tmp['X_test']
    y_test = tmp['y_test']
    return X_train, y_train, X_test, y_test

def choose_parameters_XGBoost(X_train, y_train, timeout=True, timeout_seconds=60):
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=random_state)

    parameters_acc_list = []

    start_time = time.time()  # Record start time

    total_params = 8*2*2*2*2*22*19*19*19*2

    for learning_rate in np.logspace(-7, 0, 8):
        for max_depth in [1, 10]:
            for subsample in [0.2, 1]:
                for colsample_bytree in [0.2, 1]:
                    for colsample_bylevel in [0.2, 1]:
                        for n_estimators in [100, 4000]:
                            parameters = {'learning_rate': learning_rate,
                                            'max_depth': max_depth,
                                            'subsample': subsample,
                                            'colsample_bytree': colsample_bytree,
                                            'colsample_bylevel': colsample_bylevel,
                                            'n_estimators': n_estimators,
                                            'verbosity': 0,
                                            'random_state': random_state}
                            model = XGBClassifier(**parameters)
                            model.fit(X_train, y_train)
                            acc_val = accuracy_score(y_val, model.predict(X_val))
                            parameters_acc_list.append((parameters, acc_val))

                            # Check timeout
                            elapsed_time = time.time() - start_time
                            if timeout and elapsed_time > timeout_seconds:
                                # Find the tuple with the maximum accuracy
                                best_parameters, best_accuracy = max(parameters_acc_list, key=lambda x: x[1])

                                end_time = time.time()  # Record end time
                                elapsed_time = end_time - start_time

                                print(f"Number of parameters processed: {len(parameters_acc_list)}/{total_params}")
                                print("Timeout reached. Returning best parameters found so far.")
                                print("Best Parameters:", best_parameters)
                                print("Best Accuracy:", best_accuracy)
                                print(f"Time elapsed to find Best Parameters (Timeout): {elapsed_time}s")
                                return max(parameters_acc_list, key=lambda x: x[1])[0]

    # Find the tuple with the maximum accuracy
    best_parameters, best_accuracy = max(parameters_acc_list, key=lambda x: x[1])

    end_time = time.time()  # Record end time
    elapsed_time = end_time - start_time

    print("Best Parameters:", best_parameters)
    print("Best Accuracy:", best_accuracy)
    print(f"Time elapsed to find Best Parameters: {elapsed_time}s")

    return best_parameters

def choose_parameters_LightGBoost(X_train, y_train, timeout=True, timeout_seconds=60):
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=random_state)

    parameters_acc_list = []
    total_params = 46*18*5*2*9*9*8*2

    start_time = time.time()  # Record start time

    for num_leaves in range(5,50):
        for max_depth in range(3, 20):
            for learning_rate in np.logspace(-3, 0, 4):
                for n_estimators in [50, 2000]:
                    for min_child_weight in np.logspace(-5, 4, 9):
                        for reg_alpha in  [0, 1e-1, 1, 2, 5, 7, 10, 50, 100]:
                            for reg_lambda in  [0, 1e-1, 1, 5, 10, 20, 50, 100]:
                                for subsample in  [0.2, 0.8]:
                                    parameters = {'num_leaves': num_leaves,
                                                    'max_depth': max_depth,
                                                    'learning_rate': learning_rate,
                                                    'n_estimators': n_estimators,
                                                    'min_child_weight': min_child_weight,
                                                    'reg_alpha': reg_alpha,
                                                    'reg_lambda': reg_lambda,
                                                    'subsample': subsample,
                                                    'verbosity': -1,
                                                    'random_state': random_state}
                                    # Suppress LightGBM warnings
                                    with warnings.catch_warnings():
                                        warnings.simplefilter("ignore")
                                        lgb_train = LGBMClassifier(**parameters).fit(X_train, y_train)
                                        y_pred = lgb_train.predict(X_val)
                                        acc_val = accuracy_score(y_val, y_pred)
                                        parameters_acc_list.append((parameters, acc_val))

                                    # Check timeout
                                    elapsed_time = time.time() - start_time
                                    if timeout and elapsed_time > timeout_seconds:
                                        # Find the tuple with the maximum accuracy
                                        best_parameters, best_accuracy = max(parameters_acc_list, key=lambda x: x[1])

                                        end_time = time.time()  # Record end time
                                        elapsed_time = end_time - start_time

                                        print(f"Number of parameters processed: {len(parameters_acc_list)}/{total_params}")
                                        print("Timeout reached. Returning best parameters found so far.")
                                        print("Best Parameters:", best_parameters)
                                        print("Best Accuracy:", best_accuracy)
                                        print(f"Time elapsed to find Best Parameters (Timeout): {elapsed_time}s")
                                        return max(parameters_acc_list, key=lambda x: x[1])[0]

    # Find the tuple with the maximum accuracy
    best_parameters, best_accuracy = max(parameters_acc_list, key=lambda x: x[1])

    end_time = time.time()  # Record end time
    elapsed_time = end_time - start_time

    print("Best Parameters:", best_parameters)
    print("Best Accuracy:", best_accuracy)
    print(f"Time elapsed to find Best Parameters: {elapsed_time}s")

    return best_parameters

def choose_parameters_CatBoost(X_train, y_train, timeout=True, timeout_seconds=60):
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=random_state)

    parameters_acc_list = []
    total_params = 6*2*10*10*2*2

    start_time = time.time()  # Record start time

    for learning_rate in np.logspace(-5, 0, 6):
        for random_strength in [1, 20]:
            for l2_leaf_reg in np.linspace(0, 10.0, 10):
                for bagging_temperature in np.linspace(0, 1.0, 10):
                    for leaf_estimation_iterations in [1, 20]:
                        for iterations in  [100, 4000]:
                            parameters = {'learning_rate': learning_rate,
                                            'random_strength': random_strength,
                                            'l2_leaf_reg': l2_leaf_reg,
                                            'bagging_temperature': bagging_temperature,
                                            'leaf_estimation_iterations': leaf_estimation_iterations,
                                            'iterations': iterations,
                                            'verbose': 0,
                                            'random_state': random_state}
                            # Suppress LightGBM warnings
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                cat_model = CatBoostClassifier(**parameters).fit(X_train, y_train)
                                y_pred = cat_model.predict(X_val)
                                acc_val = accuracy_score(y_val, y_pred)
                                parameters_acc_list.append((parameters, acc_val))

                            # Check timeout
                            elapsed_time = time.time() - start_time
                            if timeout and elapsed_time > timeout_seconds:
                                # Find the tuple with the maximum accuracy
                                best_parameters, best_accuracy = max(parameters_acc_list, key=lambda x: x[1])

                                end_time = time.time()  # Record end time
                                elapsed_time = end_time - start_time

                                print(f"Number of parameters processed: {len(parameters_acc_list)}/{total_params}")
                                print("Timeout reached. Returning best parameters found so far.")
                                print("Best Parameters:", best_parameters)
                                print("Best Accuracy:", best_accuracy)
                                print(f"Time elapsed to find Best Parameters (Timeout): {elapsed_time}s")
                                return max(parameters_acc_list, key=lambda x: x[1])[0]

    # Find the tuple with the maximum accuracy
    best_parameters, best_accuracy = max(parameters_acc_list, key=lambda x: x[1])

    end_time = time.time()  # Record end time
    elapsed_time = end_time - start_time

    print("Best Parameters:", best_parameters)
    print("Best Accuracy:", best_accuracy)
    print(f"Time elapsed to find Best Parameters: {elapsed_time}s")

    return best_parameters

def run_XGBoost(file_path, X_train, y_train, X_test, y_test, default=True, timeout=True, timeout_seconds=60):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if default:
        best_parameters = {'random_state': random_state}
    else:
        best_parameters = choose_parameters_XGBoost(X_train, y_train, timeout=timeout, timeout_seconds=timeout_seconds)

    model = XGBClassifier(**best_parameters)

    start_time = time.time()  # Record start time

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc_test = accuracy_score(y_test, y_pred)

    end_time = time.time()  # Record end time
    elapsed_time = end_time - start_time

    file_name = os.path.basename(file_path)

    print(f'Accuracy on {file_name} (XGBoost): {acc_test}')
    print(f'Time elapsed for {file_name} (XGBoost): {elapsed_time}s\n')

def run_LightGBoost(file_path, X_train, y_train, X_test, y_test, default=True, timeout=True, timeout_seconds=60):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    start_time = time.time()  # Record start time

    # Suppress LightGBM warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if default:
            best_parameters = {'verbosity': -1, 'random_state': random_state}
        else:
            best_parameters = choose_parameters_LightGBoost(X_train, y_train, timeout=timeout, timeout_seconds=timeout_seconds)

        lgb_train = LGBMClassifier(**best_parameters).fit(X_train, y_train)
        y_pred = lgb_train.predict(X_test)
        acc_test = accuracy_score(y_test, y_pred)

        end_time = time.time()  # Record end time
        elapsed_time = end_time - start_time

        file_name = os.path.basename(file_path)

        print(f'Accuracy on {file_name} (LightXGBoost): {acc_test}')
        print(f'Time elapsed for {file_name} (LightXGBoost): {elapsed_time}s\n')

def run_CatBoost(file_path, X_train, y_train, X_test, y_test, default=True, timeout=True, timeout_seconds=60):
    n_classes = len(np.unique(y_train))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    start_time = time.time()  # Record start time

    # Suppress CatBoost output
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if default:
            best_parameters = {'verbose': 0, 'random_state': random_state}
        else:
            best_parameters = choose_parameters_CatBoost(X_train, y_train, timeout=timeout, timeout_seconds=timeout_seconds)

        cat_model = CatBoostClassifier(**best_parameters).fit(X_train, y_train)
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
        run_LightGBoost(file_path, X_train, y_train, X_test, y_test)

        # Run CatBoost
        run_CatBoost(file_path, X_train, y_train, X_test, y_test)
