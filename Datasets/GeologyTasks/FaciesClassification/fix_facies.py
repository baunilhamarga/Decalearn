import numpy as np

if __name__ == '__main__':
    tmp = np.load('FaciesClassificationYananGasField.npz')
    X_train, y_train = tmp['X_train'], tmp['y_train']-1
    X_test, y_test = tmp['X_test'], tmp['y_test']-1
    np.savez_compressed('FaciesClassificationYananGasField.npz', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
