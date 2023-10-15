import numpy as np
import tensorflow as tf
import random as rn
import numpy as np

import os

from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import LSTM
from keras.optimizers import Adam
from keras import regularizers
from keras.constraints import max_norm
from math import sqrt
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
from numpy import concatenate
import warnings
warnings.filterwarnings('ignore')


def parser(x):
	return datetime.strptime(x, '%Y%m')

# Create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

# Invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# Scale train and test data to [-1, 1]
def scale(train, test):
	# Fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	# Transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# Transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled

# Inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = np.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]

# Convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataset = np.insert(dataset,[0]*look_back,0)    
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset[i + look_back])
	dataY= np.array(dataY)        
	dataY = np.reshape(dataY,(dataY.shape[0],1))
	dataset = np.concatenate((dataX,dataY),axis=1)  
	return dataset

def generate_npz(series, look_back=5, file_name="cumulative_oil(indian)", stationary=False, scaled=False, check_content=False):
	raw_values = series.values

	# Transform data to be stationary or use raw values
	if stationary:
		data = difference(raw_values, 1).values
		file_name += "_stationary"
	else:
		data = raw_values

	# Create dataset
	dataset = create_dataset(data, look_back)

	# Split into train and test sets
	train_size = int(dataset.shape[0] * 0.7)
	test_size = dataset.shape[0] - train_size
	train, test = dataset[0:train_size], dataset[train_size:]

	# Transform the scale of the data
	if scaled:
		scaler, train, test = scale(train, test)
		file_name += "_scaled"

	# Divide in X, y
	X_train, y_train = np.array([row[:-1] for row in train]), np.array([row[-1] for row in train])
	X_test, y_test = np.array([row[:-1] for row in test]), np.array([row[-1] for row in test])

	# Export .npz
	np.savez_compressed(file_name, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

	# Check the saved arrays shape and content
	if check_content:
		print(f"train.shape:{train.shape}")
		print(train)
		print(f"X_train.shape:{X_train.shape}")
		print(X_train)
		print(f"y_train.shape:{y_train.shape}")
		print(y_train)
		print(f"test.shape:{test.shape}")
		print(test)
		print(f"X_test.shape:{X_test.shape}")
		print(X_test)
		print(f"y_test.shape:{y_test.shape}")
		print(y_test)

if __name__ == '__main__':
	# Load dataset
	series = read_csv("cumulative_oil(indian).csv", header=0,parse_dates=[0],index_col=0, squeeze=True)
	generate_npz(series)