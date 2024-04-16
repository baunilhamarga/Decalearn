from autogluon.tabular import TabularPredictor
import numpy as np
import pandas as pd
import argparse

if __name__ == "__main__":
    random_state = 12227  # AutoGluon doesn't let us change its seed, it uses seed 0 at every opportunity

    np.random.seed(random_state)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Facies")
    parser.add_argument("--timeout", default=1 / 120)  # in hours

    args = parser.parse_args()
    dataset = args.dataset
    timeout = float(args.timeout)

    tmp = np.load("../Data/Classification/" + dataset + ".npz")

    X_train = pd.DataFrame(tmp["X_train"])
    y_train = pd.DataFrame(tmp["y_train"], columns=["target"])
    X_test = pd.DataFrame(tmp["X_test"])
    y_test = pd.DataFrame(tmp["y_test"], columns=["target"])

    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    label = "target"
    time_limit = int(timeout * 3600)  # convert from seconds to hours
    metric = "accuracy"  # evaluation metric
    path = f"AutoGluon/{dataset}_{timeout}h"
    # Recommended settings for maximizing predictive performance
    predictor = TabularPredictor(
        label, eval_metric=metric, path=path, log_to_file=True
    ).fit(train_data, time_limit=time_limit, presets="best_quality")

    # Load the predictor by specifying the path it is saved to on disk.
    # predictor = TabularPredictor.load("Autogluon/{dataset}_{timeout}h")

    y_pred = predictor.predict(test_data)

    print(predictor.evaluate(test_data))

    leaderboard = predictor.leaderboard(test_data)
    print(leaderboard)
    leaderboard.to_csv(f"AutoGluon/leaderboard_{dataset}_{timeout}.csv", index=False)
