from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import time
import os

from tabpfn import TabPFNClassifier

def load_data(file_path):
    tmp = np.load(file_path)
    X_train = tmp['X_train']
    y_train = tmp['y_train']
    X_test = tmp['X_test']
    y_test = tmp['y_test']
    return X_train, y_train, X_test, y_test

def run_tabpfn(file_path, X_train, y_train, X_test, y_test):
    # N_ensemble_configurations controls the number of model predictions that are ensembled with feature and class rotations (See our work for details).
    # When N_ensemble_configurations > #features * #classes, no further averaging is applied.

    start_time = time.time()  # Record start time

    classifier = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)

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
        # dataset too big for tabpfn'/home/baunilha/Repositories/Decalearn/Datasets/GeologyTasks/FaciesClassification/FaciesClassificationYananGasField.npz',
    ]
    for file_path in file_paths:
        X_train, y_train, X_test, y_test = load_data(file_path)
        run_tabpfn(file_path, X_train, y_train, X_test, y_test)

