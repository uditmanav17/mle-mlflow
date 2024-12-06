from pathlib import Path

import mlflow
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    jaccard_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    zero_one_loss,
)


def classification_metrics(df: pd.DataFrame):
    metrics = {}
    metrics["accuracy_score"] = accuracy_score(df["y_pred"], df["y_test"])
    metrics["average_precision_score"] = average_precision_score(
        df["y_pred"], df["y_test"]
    )
    metrics["f1_score"] = f1_score(df["y_pred"], df["y_test"])
    metrics["jaccard_score"] = jaccard_score(df["y_pred"], df["y_test"])
    metrics["log_loss"] = log_loss(df["y_pred"], df["y_test"])
    metrics["matthews_corrcoef"] = matthews_corrcoef(df["y_pred"], df["y_test"])
    metrics["precision_score"] = precision_score(df["y_pred"], df["y_test"])
    metrics["recall_score"] = recall_score(df["y_pred"], df["y_test"])
    metrics["zero_one_loss"] = zero_one_loss(df["y_pred"], df["y_test"])
    return metrics


if __name__ == "__main__":
    with mlflow.start_run(run_name="evaluate_model") as run:
        mlflow.set_tag("mlflow.runName", "evaluate_model")
        preds_path = Path("data/predictions/test_predictions.csv")
        preds_path.parent.mkdir(exist_ok=True, parents=True)
        df = pd.read_csv(preds_path)
        metrics = classification_metrics(df)

        mlflow.log_metrics(metrics)
