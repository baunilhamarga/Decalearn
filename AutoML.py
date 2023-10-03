import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.decomposition import PCA
from pprint import pprint
import autosklearn.classification


from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers

if __name__ == '__main__':
    random_state = 12227

    tmp = np.load('<>.npz')

    X_train = tmp['x_train']
    y_train = tmp['y_train']
    X_test = tmp['x_test']
    y_test = tmp['y_test']

    n_classes = len(np.unique(y_train))

    include = {
    'classifier': ["adaboost", "decision_tree", "extra_trees", "gradient_boosting", "lda", "qda", "random_forest", "sgd"],
    'feature_preprocessor': ["no_preprocessing"]
    }

    cls = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=54000, include=include)
    cls.fit(X_train, y_train, X_test=X_val, y_test=y_val)

    X_train, y_train = np.concatenate((X_train, X_val), axis=0), np.concatenate((y_train, y_val), axis=0)
    cls.refit(X_train, y_train)

    y_pred = cls.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Dataset [{}] Accuracy score [{}]".format(dataset_filepath.split('/')[-1], acc))
    pprint(cls.show_models(), indent=4)