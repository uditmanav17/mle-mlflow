import datetime
import warnings

import mlflow.sklearn
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


def acquire_training_data():
    # yfin.pdr_override()
    start = datetime.datetime(2019, 7, 1)
    end = datetime.datetime(2019, 9, 30)
    df = yf.download("BTC-USD", start=start, end=end)
    return df


def digitize(n):
    return 1 if n > 0 else 0


def rolling_window(a, window):
    """
    Takes np.array 'a' and size 'window' as parameters
    Outputs an np.array with all the ordered sequences of values of 'a' of size 'window'
    e.g. Input: ( np.array([1, 2, 3, 4, 5, 6]), 4 )
         Output:
                 array([[1, 2, 3, 4],
                       [2, 3, 4, 5],
                       [3, 4, 5, 6]])
    """
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def prepare_training_data(data):
    """
    Return a prepared numpy dataframe
    input : Dataframe with expected schema

    """
    data["Delta"] = data["Close"] - data["Open"]
    data["to_predict"] = data["Delta"].apply(lambda d: digitize(d))
    return data


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    with mlflow.start_run():
        training_data = acquire_training_data()
        # training_data = pd.read_csv("test.csv")

        prepared_training_data_df = prepare_training_data(training_data)

        btc_mat = prepared_training_data_df.to_numpy()

        WINDOW_SIZE = 14

        X = rolling_window(btc_mat[:, 7], WINDOW_SIZE)[:-1, :]
        Y = prepared_training_data_df["to_predict"].to_numpy()[WINDOW_SIZE:]

        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.25, random_state=4284, stratify=Y
        )

        clf = RandomForestClassifier(
            bootstrap=True,
            criterion="gini",
            min_samples_split=2,
            min_weight_fraction_leaf=0.0,
            n_estimators=50,
            random_state=4284,
            verbose=0,
        )

        clf.fit(X_train, y_train)

        predicted = clf.predict(X_test)

        mlflow.sklearn.log_model(clf, "model_random_forest")

        print(classification_report(y_test, predicted))
        # fmt: off
        mlflow.log_metric("precision_label_0", precision_score(y_test, predicted, pos_label=0))
        mlflow.log_metric("recall_label_0", recall_score(y_test, predicted, pos_label=0))
        mlflow.log_metric("f1score_label_0", f1_score(y_test, predicted, pos_label=0))

        mlflow.log_metric("precision_label_1", precision_score(y_test, predicted, pos_label=1))
        mlflow.log_metric("recall_label_1", recall_score(y_test, predicted, pos_label=1))
        mlflow.log_metric("f1score_label_1", f1_score(y_test, predicted, pos_label=1))
        # fmt: on
