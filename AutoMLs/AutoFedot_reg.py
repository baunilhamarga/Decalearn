aimport numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from pprint import pprint
from fedot.api.main import Fedot

if __name__ == '__main__':
    random_state = 12227

    tmp = np.load('../Datasets/Nasa Predictive Maintenance (RUL)/data_nasa.npz')

    X_train = tmp['X_train']
    y_train = tmp['y_train']
    X_test = tmp['X_test']
    y_test = tmp['y_test']

    # Use regression problem
    reg = Fedot(problem='regression', timeout=5, n_jobs=-1)
    reg.fit(features=X_train, target=y_train)
    prediction = reg.predict(features=X_test)

    # Calculate mean squared error (MSE) as the evaluation metric for regression
    mse = mean_squared_error(y_test, prediction)

    print("Mean Squared Error: {:.4f}".format(mse))
