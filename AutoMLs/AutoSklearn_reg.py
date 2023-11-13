import numpy as np
import autosklearn.regression  # Import AutoSklearnRegressor for regression
from sklearn.metrics import mean_absolute_error, mean_squared_error


if __name__ == '__main__':
    random_state = 12227

    tmp = np.load('/home/baunilha/Repositories/Decalearn/Datasets/Nasa Predictive Maintenance (RUL)/data_nasa.npz')

    X_train = tmp['X_train']
    y_train = tmp['y_train']
    X_test = tmp['X_test']
    y_test = tmp['y_test']
    X_val = tmp['X_val']
    y_val = tmp['y_val']

    y_val = y_val.flatten()

    include = {
        'regressor': ["adaboost", "decision_tree", "extra_trees", "gradient_boosting", "random_forest", "sgd"],
        'feature_preprocessor': ["no_preprocessing"]
    }

    cls = autosklearn.regression.AutoSklearnRegressor(time_left_for_this_task=54000, include=include)
    cls.fit(X_train, y_train, X_test=X_val, y_test=y_val)

    X_train, y_train = np.concatenate((X_train, X_val), axis=0), np.concatenate((y_train, y_val), axis=0)
    cls.refit(X_train, y_train)

    y_pred = cls.predict(X_test)

    # Use regression metrics like Mean Absolute Error (MAE) or Mean Squared Error (MSE)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print("Mean Absolute Error: {:.4f}".format(mae))
    print("Mean Squared Error: {:.4f}".format(mse))
