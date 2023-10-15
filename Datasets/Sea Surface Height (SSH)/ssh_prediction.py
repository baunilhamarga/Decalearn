import numpy as np
import random
from sklearn.metrics import mean_absolute_error
import copy
from sklearn.preprocessing import StandardScaler

from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers


class PositionalEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(PositionalEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(units=self.projection_dim)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, position):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(self.projection_dim)[np.newaxis, :],
                                     self.projection_dim)

        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        position_encoding = self.positional_encoding(self.num_patches)
        encoded = self.projection(patch) + position_encoding
        return encoded

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'num_patches': self.num_patches,
            'projection_dim': self.projection_dim
        })
        return config

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(units=self.projection_dim)

        # if weights is not None:
        #     self.projection = layers.Dense(units=projection_dim, weights=weights)

        self.position_embedding = layers.Embedding(
            input_dim=num_patches,
            output_dim=self.projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'num_patches': self.num_patches,
            'projection_dim': self.projection_dim
        })
        return config

def Transformer(input_shape, projection_dim, num_heads):

    inputs = layers.Input(shape=input_shape)
    encoded_patches = PatchEncoder(input_shape[0], projection_dim)(inputs)
    #encoded_patches = PositionalEncoder(input_shape[0], projection_dim)(inputs)

    num_transformer_blocks = len(num_heads)
    for i in range(num_transformer_blocks):

        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(num_heads=num_heads[i], key_dim=projection_dim, dropout=0.0)(x1, x1)

        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)

        # MLP Size of the transformer layers
        transformer_units = [projection_dim * 2, projection_dim]

        x3 = layers.Dense(projection_dim * 2, activation=tf.nn.gelu)(x3)
        x3 = layers.Dense(projection_dim, activation=tf.nn.gelu)(x3)

        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])


    encoded_patches = layers.Flatten()(encoded_patches)
    outputs = layers.Dense(1)(encoded_patches)

    return keras.Model(inputs, outputs)

def temporal_window(X, look_back=1):
    X = np.insert(X,[0]*look_back,0)
    observations, targets = [], []
    for i in range(len(X)-look_back):
        a = X[i:(i+look_back)]
        observations.append(a)
        targets.append(X[i + look_back])

    observations = np.array(observations)
    targets = np.array(targets)
    return observations, targets

def IoA(observed, predicted):
    mean_observed = np.mean(observed)

    numerator = np.sum((observed - predicted) ** 2)
    denominator = np.sum((np.abs(predicted - mean_observed) + np.abs(observed - mean_observed)) ** 2)

    return 1 - (numerator / denominator)

if __name__ == '__main__':
    random_state = 12227

    tmp = np.load('ssh_praticagem.npz')
    X_train = tmp['X_train']
    X_test = tmp['X_test']

    X_train, y_train = temporal_window(X_train, look_back=3)
    X_test, y_test = temporal_window(X_test, look_back=3)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    projection_dim = 128
    num_heads = [16, 8, 4]
    transformer_units = [projection_dim * 2, projection_dim]


    #It expands (samples, windows) to (samples, 1 feature, windows)
    X_train = np.expand_dims(X_train, axis=1)
    X_test = np.expand_dims(X_test, axis=1)

    input_shape = (X_train.shape[1:])
    model = Transformer(input_shape, projection_dim, num_heads)

    model.compile(loss='mean_squared_error', optimizer='Adam')
    model.fit(X_train, y_train, epochs=2, batch_size=512, verbose=2)

    y_pred = model.predict(X_test)

    MAE = mean_absolute_error(y_test, y_pred)

    print('MAE [{:.4f}] IoA [{:.4f}] '.format(MAE, IoA(y_pred[:, 0], y_test)))