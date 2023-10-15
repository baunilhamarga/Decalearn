import numpy as np
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import layers

def res_block(inputs, norm_type, activation, dropout, projection_dim):
  """Residual block of MLP."""

  norm = (
      layers.LayerNormalization
      if norm_type == 'L'
      else layers.BatchNormalization
  )

  x = norm()(inputs)
  x = layers.Dense(projection_dim, activation=activation)(x)
  x = layers.Dropout(dropout)(x)
  x = layers.Dense(inputs.shape[-1], activation=activation)(x)
  res = x + inputs

  return res


def build_model(
    input_shape,
    n_classes,
    norm_type='L',
    activation='relu',
    n_block=3,
    dropout=0,
    projection_dim=128):
  """Build MLP-Residual model."""

  inputs = tf.keras.Input(shape=input_shape)
  x = inputs
  for _ in range(n_block):
    x = res_block(x, norm_type, activation, dropout, projection_dim)

  x = layers.Flatten()(x)
  outputs = layers.Dense(n_classes, activation='softmax')(x)

  return tf.keras.Model(inputs, outputs)

if __name__ == '__main__':
    np.random.seed(12227)

    tmp = np.load('data/UTD-MHAD2_1s.npz')
    X_train, X_test, y_train, y_test = tmp['X_train'], tmp['X_test'], tmp['y_train'], tmp['y_test']
    n_classes = len(np.unique(y_train))

    le = LabelBinarizer()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)

    input_shape = (X_train.shape[-1], 1)
    model = build_model(input_shape=input_shape,n_classes=n_classes, activation='relu',
                        n_block=6, dropout=0.05, projection_dim=16)

    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=2)

    y_pred = model.predict(X_test)
    acc_test = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    print(acc_test)