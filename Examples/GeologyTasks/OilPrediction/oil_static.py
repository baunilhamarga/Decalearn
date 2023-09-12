import numpy as np
import random as rn
import argparse


from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import layers
from math import sqrt
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
from numpy import concatenate

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches,
            output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

def FFN(x, hidden_units):
    #Section 3.3 in "Attention in All You Need"
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
    return x

def build_model(input_shape, projection_dim, num_heads, num_transformer_blocks):

    inputs = layers.Input(shape=input_shape)
    encoded_patches = PatchEncoder(input_shape[0], projection_dim)(inputs)

    for _ in range(num_transformer_blocks):

        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.0)(x1, x1)

        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.

        # Size of the transformer layers
        transformer_units = [projection_dim * 2, projection_dim]

        x3 = FFN(x3, hidden_units=transformer_units)

        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    encoded_patches = layers.Flatten()(encoded_patches)
    outputs = layers.Dense(1)(encoded_patches)
    #outputs = layers.Dense(n_classes, activation='softmax')(encoded_patches)

    return keras.Model(inputs, outputs)

def parser(x):
	return datetime.strptime(x, '%Y%m')

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# scale train and test data to [-1, 1]
def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = np.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]


# make a one-step forecast
def forecast(model, batch_size, X):
	X = X.reshape(1, len(X), 1)
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]

# convert an array of values into a dataset matrix
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


# compute RMSPE
def RMSPE(x,y):
	result=0
	for i in range(len(x)):
		result += ((y[i]-x[i])/x[i])**2
	result /= len(x)
	result = sqrt(result)
	result *= 100
	return result

if __name__ == '__main__':
	np.random.seed(12227)
	tf.random.set_seed(12227)
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', type=str, default='RNN')

	args = parser.parse_args()
	method = args.c

	print(args)
	#We employ the same conf. as suggested in the original paper
	if method == 'RNN':
		#See Table 4
		look_back = 5
		ep = 1551

	if method == 'GRU':
		#See Table 5
		look_back = 6
		ep = 1514

	if method == 'LSTM':
		#See Table 1
		look_back = 5
		ep = 800

	if method == 'TRAN':
		ep = 50
		look_back = 3

	#load dataset
	series = read_csv('Chinese_cumulative_oil.csv', header=0,parse_dates=[0],index_col=0, squeeze=True)

	raw_values = series.values
	# transform data to be stationary
	diff = difference(raw_values, 1)

	# create dataset x,y
	dataset = diff.values
	dataset = create_dataset(dataset, look_back)

	# split into train and test sets
	train_size = int(dataset.shape[0] * 0.7)
	test_size = dataset.shape[0] - train_size
	train, test = dataset[0:train_size], dataset[train_size:]

	# transform the scale of the data
	scaler, train_scaled, test_scaled = scale(train, test)

	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], X.shape[1], 1)

	input_shape = (X.shape[1], X.shape[2])

	if method == 'TRAN':
		#Best configuration based on validation set: Heads [2] #TransBlocks [1] dk [16] Window [3]  RMSE[0.0275]
		model = build_model(input_shape, projection_dim=16, num_heads=2,
							num_transformer_blocks=1)
	else:
		model = Sequential()

		if method == 'RNN':
			model.add(SimpleRNN(2, input_shape=input_shape, return_sequences=True))
			model.add(SimpleRNN(4, input_shape=input_shape, return_sequences=False))

		if method == 'GRU':
			model.add(GRU(5, input_shape=input_shape, return_sequences=True))
			#model.add(GRU(3, return_sequences=True))
			model.add(GRU(4, input_shape=input_shape, return_sequences=False))

		if method == 'LSTM':
			model.add(LSTM(5, input_shape=input_shape, return_sequences=True))
			model.add(LSTM(4, return_sequences=True))
			model.add(LSTM(2, input_shape=input_shape, return_sequences=False))

	model.compile(loss='mean_squared_error', optimizer='adam')

	model.fit(X, y, epochs=ep, batch_size=32, verbose=1)

	# forecast the entire training dataset to build up state for forecasting
	print('Forecasting Training Data')
	predictions_train = list()
	for i in range(len(train_scaled)):
		# make one-step forecast
		X, y = train_scaled[i, 0:-1], train_scaled[i, -1]
		yhat = forecast(model, 1, X)
		# invert scaling
		yhat = invert_scale(scaler, X, yhat)
		# invert differencing
		yhat = inverse_difference(raw_values, yhat, len(raw_values) - i)
		# store forecast
		predictions_train.append(yhat)
		expected = raw_values[i + 1]
		print('Month=%d, Predicted=%.5f, Expected=%.5f' % (i + 1, yhat, expected))

	# report performance
	rmse_train = sqrt(mean_squared_error(raw_values[1:len(train_scaled) + 1], predictions_train))
	print('Train RMSE: %.3f' % rmse_train)
	# report performance using RMSPE
	rmspe_train = RMSPE(raw_values[1:len(train_scaled) + 1], predictions_train)
	print('Train RMSPE: %.3f' % rmspe_train)

	# forecast the test data
	print('Forecasting Testing Data')
	predictions_test = list()
	for i in range(len(test_scaled)):
		# make one-step forecast
		X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
		yhat = forecast(model, 1, X)
		# invert scaling
		yhat = invert_scale(scaler, X, yhat)
		# invert differencing
		yhat = inverse_difference(raw_values, yhat, len(test_scaled) + 1 - i)
		# store forecast
		predictions_test.append(yhat)
		expected = raw_values[len(train) + i + 1]
		print('Month=%d, Predicted=%.5f, Expected=%.5f' % (i + 1, yhat, expected))

	# report performance using RMSE
	rmse_test = sqrt(mean_squared_error(raw_values[-len(test_scaled):], predictions_test))
	print('Test RMSE: %.3f' % rmse_test)
	# report performance using RMSPE
	rmspe_test = RMSPE(raw_values[-len(test_scaled):], predictions_test)
	print('Test RMSPE: %.3f' % rmspe_test)

	predictions = np.concatenate((predictions_train, predictions_test), axis=0)

	# line plot of observed vs predicted
	fig, ax = plt.subplots(1)
	ax.plot(raw_values, label='original', color='blue')
	ax.plot(predictions, label='predictions', color='red')
	ax.axvline(x=len(train_scaled) + 1, color='k', linestyle='--')
	ax.legend(loc='upper left')
	ax.set_xlabel("Months", fontsize=16)
	ax.set_ylabel("cumulative oil production", fontsize=16)
	plt.show()