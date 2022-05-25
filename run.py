from pathlib import Path

import openml
from autosklearn.classification import AutoSklearnClassifier

dionisis = 168355
christine = 168908

out = Path(__file__).parent.resolve() / "output"

config = {
    "time_left_for_this_task": 60 * 60,
    "seed": 1,
    "memory_limit": 8_000,
    "n_jobs": 2,
    "delete_tmp_folder_after_terminate": False,
}


if __name__ == "__main__":
    askl = AutoSklearnClassifier(**config)

    task = openml.tasks.get_task(christine)
    X, y = task.get_X_and_y(dataset_format="dataframe")
    train_indices, test_indices = task.get_train_test_split_indices(
        repeat=0,
        fold=0,
        sample=0,
    )

    X_train = X.iloc[train_indices]
    y_train = y.iloc[train_indices]
    print(X.shape, y.shape)
    # X_test = X[test_indices]
    # y_test = y.iloc[test_indices]

    askl.fit(X=X_train, y=y_train, dataset_name="christine")

    print("----------------------------------------")
    print(askl.automl_._backend.temporary_directory)
    print("----------------------------------------")
