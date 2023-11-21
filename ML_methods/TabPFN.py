from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import time
import os
from itertools import product

from tabpfn import TabPFNClassifier

import sys
sys.path.append('..')
from Scripts import generate_coreset

random_state = 12227

def load_data(file_path, use_coreset=False, coreset_size=1024):
    tmp = np.load(file_path)
    if use_coreset and len(tmp['X_train']) > coreset_size:
        X_train, y_train = generate_coreset.reconstruct_coreset(tmp['X_train'], tmp['y_train'], coreset_size)
    else:
        X_train = tmp['X_train']
        y_train = tmp['y_train']
    X_test = tmp['X_test']
    y_test = tmp['y_test']
    return X_train, y_train, X_test, y_test


def choose_parameters_TabPFN(X_train, y_train, timeout=True, timeout_seconds=60, verbose=False):
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=random_state)

    parameters_acc_list = []
    
    tested_params = {
                        'device': ['cpu'],
                        'N_ensemble_configurations': [2, 4, 8, 16, 32, 64, 128],
                    }

    total_params = np.prod([len(values) for values in tested_params.values()])

    start_time = time.time()  # Record start time

    # Generate all combinations of parameters
    for param_values in product(*tested_params.values()):
        parameters = dict(zip(tested_params.keys(), param_values))
        model = TabPFNClassifier(**parameters)
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

            print(f"\nNumber of parameters processed: {len(parameters_acc_list)}/{total_params}")
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


def run_tabpfn(file_path, X_train, y_train, X_test, y_test, default=True, timeout=True, timeout_seconds=60, verbose=False):
    # N_ensemble_configurations controls the number of model predictions that are ensembled with feature and class rotations (See our work for details).
    # When N_ensemble_configurations > #features * #classes, no further averaging is applied.

    if default:
        best_parameters = {'device': 'cpu', 'N_ensemble_configurations': 32}
    else:
        best_parameters = choose_parameters_TabPFN(X_train, y_train, timeout=timeout, timeout_seconds=timeout_seconds, verbose=verbose)

    start_time = time.time()  # Record start time

    classifier = TabPFNClassifier(**best_parameters)
    
    classifier.fit(X_train, y_train)
    y_eval, p_eval = classifier.predict(X_test, return_winning_probability=True)
    acc_test = accuracy_score(y_test, y_eval)


    end_time = time.time()  # Record end time

    elapsed_time = end_time - start_time

    file_name = os.path.basename(file_path)

    print(f'Accuracy on {file_name}: {acc_test}')
    print(f'Time elapsed for {file_name}: {elapsed_time}s\n')


if __name__ == '__main__':
    file_paths = [
        '/home/baunilha/Repositories/Decalearn/Datasets/Multimodal Human Action/data/UTD-MHAD2_1s.npz',
        '/home/baunilha/Repositories/Decalearn/Datasets/GeologyTasks/FaciesClassification/FaciesClassificationYananGasField.npz',
    ]
    for file_path in file_paths:
        X_train, y_train, X_test, y_test = load_data(file_path, use_coreset=True, coreset_size=1024)
        run_tabpfn(file_path, X_train, y_train, X_test, y_test, default=False, timeout=False, verbose=True, timeout_seconds=10)

