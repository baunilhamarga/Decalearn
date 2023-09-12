import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from pprint import pprint
from sklearn.preprocessing import StandardScaler

from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers


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

def Transformer(input_shape, projection_dim, num_heads, n_classes):

    inputs = layers.Input(shape=input_shape)
    encoded_patches = PatchEncoder(input_shape[0], projection_dim)(inputs)

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
    outputs = layers.Dense(n_classes, activation='softmax')(encoded_patches)

    return keras.Model(inputs, outputs)

if __name__ == '__main__':
    random_state = 12227

    tmp = np.load('osha_train_test.npz')

    X_train = tmp['X_train']
    y_train = tmp['y_train']
    X_test = tmp['X_test']
    y_test = tmp['y_test']

    indices_to_keep = np.logical_not(np.logical_or(y_train == 6, y_train == 10))
    X_train = X_train[indices_to_keep]
    y_train = y_train[indices_to_keep]

    n_classes = len(np.unique(y_train))

    le = preprocessing.LabelBinarizer()
    y_train = le.fit_transform(y_train)
    y_test = le.fit_transform(y_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    projection_dim = 128
    num_heads = [32, 16, 8, 4]
    transformer_units = [projection_dim * 2, projection_dim]


    X_train = np.expand_dims(X_train, axis=1)
    X_test = np.expand_dims(X_test, axis=1)
    input_shape = (X_train.shape[1:])
    model = Transformer(input_shape, projection_dim, num_heads, n_classes)

    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=2)

    y_pred = model.predict(X_test)

    acc_test = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))

    print('Testing Accuracy [{:.4f}]'.format(acc_test))
