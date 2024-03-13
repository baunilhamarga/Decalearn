from autogluon.tabular import TabularPredictor
import numpy as np
import pandas as pd


if __name__ == '__main__':
    random_state = 12227 # AutoGluon doesn't let us change its seed, it uses seed 0 at every opportunity

    tmp = np.load('../Datasets/GeologyTasks/FaciesClassification/FaciesClassificationYananGasField.npz')
    custom_project_name = 'AutoKeras/Facies_core'

    X_train = pd.DataFrame(tmp['X_train'])
    y_train = pd.DataFrame(tmp['y_train'], columns=['target'])
    X_test = pd.DataFrame(tmp['X_test'])
    y_test = pd.DataFrame(tmp['y_test'], columns=['target'])

    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    label = 'target'
    time_limit = 133200  # 37h in seconds
    metric = 'accuracy'  # evaluation metric
    # Recommended settings for maximizing predictive performance
    predictor = TabularPredictor(label, eval_metric=metric).fit(train_data, time_limit=time_limit, presets='best_quality')

    y_pred = predictor.predict(test_data)

    print(predictor.evaluate(test_data))

    print(predictor.leaderboard(test_data))