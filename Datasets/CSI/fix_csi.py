import numpy as np

if __name__ == '__main__':
    tmp = np.load('dataset_csi.npz')
    X_train, y_train = tmp['x_train'], tmp['y_train']
    X_test, y_test = tmp['x_test'], tmp['y_test']
    np.savez_compressed('dataset_csi.npz', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
