from autogluon.tabular import TabularPredictor
import numpy as np
import pandas as pd


if __name__ == '__main__':
    random_state = 12227 # AutoGluon doesn't let us change its seed, it uses seed 0 at every opportunity

    tmp = np.load('../Datasets/Multimodal Human Action/data/UTD-MHAD2_1s.npz')

    X_train = pd.DataFrame(tmp['X_train'])
    y_train = pd.DataFrame(tmp['y_train'], columns=['target'])
    X_test = pd.DataFrame(tmp['X_test'])
    y_test = pd.DataFrame(tmp['y_test'], columns=['target'])

    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    label = 'target'
    time_limit = 133200  # 37h in seconds
    metric = 'accuracy'  # evaluation metric
    path = 'AutoGluon/MAHD_37h'
    # Recommended settings for maximizing predictive performance
    predictor = TabularPredictor(label, eval_metric=metric, path=path, log_to_file=True).fit(train_data, time_limit=time_limit, presets='best_quality')

    # Load the predictor by specifying the path it is saved to on disk.
    #predictor = TabularPredictor.load("Autogluon/Facies_37h")

    y_pred = predictor.predict(test_data)

    print(predictor.evaluate(test_data))

    leaderboard = predictor.leaderboard(test_data)
    print(leaderboard)
    leaderboard.to_csv('leaderboard_MAHD_37h.csv', index=False)