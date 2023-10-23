import numpy as np
import random
import copy
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.decomposition import PCA
from pprint import pprint
from sklearn.preprocessing import StandardScaler

import autokeras as ak
import tensorflow as tf

if __name__ == '__main__':
    random_state = 12227

    tmp = np.load('../Datasets/GeologyTasks/OilPrediction/cumulative_oil(indian)_stationary.npz', allow_pickle=True)
    custom_project_name = 'AutoSklearn/cumulative_oil(indian)_stationary'

    X_train = tmp['X_train']
    y_train = tmp['y_train']
    X_test = tmp['X_test']
    y_test = tmp['y_test']

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = ak.StructuredDataRegressor(overwrite=True, max_trials=10, project_name=custom_project_name)

    model.fit(tf.data.Dataset.from_tensor_slices((X_train.astype(str), y_train)), epochs=40)

    y_pred = model.predict(tf.data.Dataset.from_tensor_slices((X_test).astype(str)))

    mse = mean_squared_error(y_test, y_pred)

    print('Mean Squared Error on Test Data: {:.4f}'.format(mse))
