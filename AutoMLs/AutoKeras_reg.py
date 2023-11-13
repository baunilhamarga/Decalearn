import numpy as np
import random
import copy
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.decomposition import PCA
from pprint import pprint
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import autokeras as ak
import tensorflow as tf

if __name__ == '__main__':
    random_state = 12227

    tmp = np.load('/home/baunilha/Repositories/Decalearn/Datasets/Sea Surface Height (SSH)/SSH_stationary.npz', allow_pickle=True)
    custom_project_name = 'AutoSklearn/test'

    X_train = tmp['X_train']
    y_train = tmp['y_train']
    X_test = tmp['X_test']
    y_test = tmp['y_test']

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=random_state)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)

    y_train = np.expand_dims(y_train, axis=1)
    y_test = np.expand_dims(y_test, axis=1)

    scaler = StandardScaler()
    y_train = scaler.fit_transform(y_train)
    y_test = scaler.transform(y_test)
    y_val = y_val.reshape(-1, 1)
    y_val = scaler.transform(y_val)


    model = ak.StructuredDataRegressor(overwrite=True, max_trials=10, project_name=custom_project_name, seed=random_state)

    # Convert your data to TensorFlow datasets
    train_data = tf.data.Dataset.from_tensor_slices((X_train.astype(str), y_train))
    val_data = tf.data.Dataset.from_tensor_slices((X_val.astype(str), y_val))

    model.fit(train_data, epochs=40, validation_data=val_data)

    y_pred = model.predict(tf.data.Dataset.from_tensor_slices((X_test).astype(str)))

    mse = mean_squared_error(y_test, y_pred)

    print('Mean Squared Error on Test Data: {:.4f}'.format(mse))
