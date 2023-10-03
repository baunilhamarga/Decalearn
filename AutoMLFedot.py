import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.decomposition import PCA
from pprint import pprint
import autosklearn.classification



if __name__ == '__main__':
    random_state = 12227

    tmp = np.load('<>.npz')

    X_train = tmp['X_train']
    y_train = tmp['y_train']
    X_test = tmp['X_test']
    y_test = tmp['y_test']

    n_classes = len(np.unique(y_train))

    cls = Fedot(problem='classification', timeout=5, preset='best_quality', n_jobs=-1)
    cls.fit(features=X_train, target=y_train)
    prediction = cls.predict(features=X_test)

    y_pred = cls.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Accuracy score [{}]".format(acc))