import numpy as np
from sklearn.metrics import accuracy_score
from fedot.api.main import Fedot
import pandas as pd
from fedot.core.pipelines.pipeline import Pipeline


if __name__ == '__main__':
    random_state = 12227

    file_paths = [
        '../Datasets/Multimodal Human Action/data/UTD-MHAD2_1s.npz',
        '../Datasets/GeologyTasks/FaciesClassification/FaciesClassificationYananGasField.npz',
        '../Datasets/Lucas/osha_train_test.npz',
        '../Datasets/AI_text/AI_text.npz',
        '../Datasets/CSI/dataset_csi.npz',
        '../Datasets/AI_text/AI_text.npz',
    ]

    tmp = np.load(file_paths[0])

    X_train = pd.DataFrame(tmp['X_train'])
    y_train = pd.DataFrame(tmp['y_train'], columns=['target'])
    X_test = pd.DataFrame(tmp['X_test'])
    y_test = pd.DataFrame(tmp['y_test'], columns=['target'])

    timeout = 2220 # 37h in minutes
    model = Fedot(problem='classification', timeout=timeout, preset='best_quality', n_jobs=-1, seed=random_state)
    best_pipeline = model.fit(features=X_train, target=y_train)
    y_pred = model.predict(features=X_test)

    print(model.get_metrics(target=y_test, metric_names=['acc']))
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy score [{}]".format(acc))

    path_to_save = 'AutoFedot/MAHD_37h'
    best_pipeline.save(path=path_to_save, create_subdir=True, is_datetime_in_path=True)
    #pipeline = Pipeline().load(path) # To load model