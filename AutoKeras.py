import numpy as np
import random
import copy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.decomposition import PCA
from pprint import pprint
from sklearn.preprocessing import StandardScaler

import autokeras as ak
import tensorflow as tf


if __name__ == '__main__':
    random_state = 12227

    tmp = np.load('<>.npz')

    X_train = tmp['X_train']
    y_train = tmp['y_train']
    X_test = tmp['X_test']
    y_test = tmp['y_test']

    n_classes = len(np.unique(y_train))

    le = preprocessing.LabelBinarizer()
    y_train = le.fit_transform(y_train)
    y_test = le.fit_transform(y_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    model = ak.StructuredDataClassifier(overwrite=True, max_trials=10)

    model.fit(tf.data.Dataset.from_tensor_slices((X_train.astype(str), y_train)), epochs=40)

    y_pred = model.predict(tf.data.Dataset.from_tensor_slices((X_test).astype(str)))

    acc_test = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))

    print('Testing Accuracy before Pruning [{:.4f}]'.format(acc_test))