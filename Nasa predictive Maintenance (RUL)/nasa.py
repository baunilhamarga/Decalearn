# https://www.kaggle.com/code/wassimderbel/nasa-predictive-maintenance-rul/notebook

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import sklearn
from sklearn.metrics import mean_squared_error, r2_score
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
import warnings
np.random.seed(12227)
warnings.filterwarnings('ignore')

# Importing train and validation data
# FD001 subset corresponds to HPC failure of the engine.
index_names = ['unit_number', 'time_cycles']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names = ['s_{}'.format(i+1) for i in range(0,21)]
col_names = index_names + setting_names + sensor_names

dftrain = pd.read_csv('Cmaps/train_FD001.txt',sep='\s+',header=None,index_col=False,names=col_names)
dfvalid = pd.read_csv('Cmaps/test_FD001.txt',sep='\s+',header=None,index_col=False,names=col_names)
y_valid = pd.read_csv('Cmaps/RUL_FD001.txt',sep='\s+',header=None,index_col=False,names=['RUL'])
dfvalid.shape

dftrain = pd.read_csv('Cmaps/train_FD001.txt',sep='\s+',header=None,index_col=False,names=col_names)
dfvalid = pd.read_csv('Cmaps/test_FD001.txt',sep='\s+',header=None,index_col=False,names=col_names)
y_valid = pd.read_csv('Cmaps/RUL_FD001.txt',sep='\s+',header=None,index_col=False,names=['RUL'])
dfvalid.shape

train = dftrain.copy()
valid = dfvalid.copy()

train

# Add RUL column to the data
# RUL corresponds to the remaining time cycles for each unit before it fails.
def add_RUL_column(df):
    train_grouped_by_unit = df.groupby(by='unit_number') 
    max_time_cycles = train_grouped_by_unit['time_cycles'].max() 
    merged = df.merge(max_time_cycles.to_frame(name='max_time_cycle'), left_on='unit_number',right_index=True)
    merged["RUL"] = merged["max_time_cycle"] - merged['time_cycles']
    merged = merged.drop("max_time_cycle", axis=1) 
    return merged

train = add_RUL_column(train)

train[['unit_number','RUL']]

# Dropping unnecessary features (labels and settings)
from sklearn.model_selection import train_test_split
drop_labels = index_names+setting_names
X_train=train.drop(columns=drop_labels).copy()
X_train, X_test, y_train, y_test=train_test_split(X_train,X_train['RUL'], test_size=0.3, random_state=12227)

 
# from oil_static.py
# split into train and test sets
#train_size = int(dataset.shape[0] * 0.7)
#test_size = dataset.shape[0] - train_size
#train, test = dataset[0:train_size], dataset[train_size:]


# Scaling the data
# MinMax scaler function: Transform features by scaling each feature to a given range.
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

#Droping the target variable
X_train.drop(columns=['RUL'], inplace=True)
X_test.drop(columns=['RUL'], inplace=True)

#Scaling X_train and X_test
X_train_s=scaler.fit_transform(X_train)
X_test_s=scaler.fit_transform(X_test)
#Conserve only the last occurence of each unit to match the length of y_valid
X_val = valid.groupby('unit_number').last().reset_index().drop(columns=drop_labels)
#scaling X_val
X_val_s=scaler.fit_transform(X_val)

np.savez_compressed('data_nasa', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, X_val=X_val)
#data = np.load('data_nasa.npz')