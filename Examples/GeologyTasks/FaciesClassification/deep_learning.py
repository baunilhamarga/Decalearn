import numpy as np
from sklearn import preprocessing
import argparse
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.metrics import accuracy_score
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

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

def build_model(input_shape, projection_dim, num_heads, num_transformer_blocks, n_classes):

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
    #outputs = layers.Dense(1)(encoded_patches)
    outputs = layers.Dense(n_classes, activation='softmax')(encoded_patches)

    return keras.Model(inputs, outputs)

def time_step_data(X,y=None, look_back=4):
    X_tmp, y_tmp = [], []
    if y is not None:
        for i in range(len(X)-look_back-1):
            a = X[i:(i+look_back)]
            X_tmp.append(a)
            y_tmp.append(y[i:(i+look_back)])
        return np.array(X_tmp), np.array(y_tmp)
    else:
        for i in range(len(X)-look_back-1):
            a = X[i:(i+look_back)]
            X_tmp.append(a)
        return np.array(X_tmp)

def to_categorical(y):
    a = y
    b = np.zeros((a.size, a.max() + 1))
    b[np.arange(a.size), a] = 1
    return b

if __name__ == '__main__':
    np.random.seed(12227)
    tf.random.set_seed(12227)

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=str, default='TRAN')

    args = parser.parse_args()
    method = args.c

    print(args)

    look_back = 2
    if method == 'TRAN':
        look_back = 1

    tmp = np.load('YananGasField.npz')
    X_train, y_train = tmp['X_train'], tmp['y_train']-1
    X_test, y_test = tmp['X_test'], tmp['y_test']-1

    n_classes = len(np.unique(y_train))
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # For temporal data
    X_train, y_train = time_step_data(X_train, y_train, look_back=look_back)
    input_shape = X_train.shape[1:]
    y_train = y_train[:, 0, :]  # np.mean(y_train, axis=1)

    if method == 'TRAN':
        model = build_model(input_shape, projection_dim=64, num_heads=4,
                            num_transformer_blocks=3, n_classes=n_classes)

    else:
        model = Sequential()

        if method == 'RNN':
            model.add(SimpleRNN(512, input_shape=(look_back, X_train.shape[-1])))
        if method == 'GRU':
            model.add(GRU(512, input_shape=(look_back, X_train.shape[-1]), return_sequences=False))
        if method == 'LSTM':
            model.add(LSTM(512, input_shape=(look_back, X_train.shape[-1]),
                           return_sequences=False))  # True when using multiple layers

        model.add(Dense(n_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=500, batch_size=256, verbose=0)

    padding = np.tile(X_test[-1], (look_back + 1, 1))  # 4 is the lookback value
    X_test = np.vstack([X_test, padding])
    X_test = time_step_data(X_test, y=None, look_back=look_back)

    y_pred = model.predict(X_test)
    acc = accuracy_score(np.argmax(y_pred, axis=1), np.argmax(y_test, axis=1))
    print('Accuracy [{:.4f}]'.format(acc))