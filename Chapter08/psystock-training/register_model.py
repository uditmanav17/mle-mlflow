import sys

import mlflow


def register_model(run_id: str):
    with mlflow.start_run(run_name="register_model") as run:
        mlflow.set_tag("mlflow.runName", "register_model")

        result = mlflow.register_model(
            f"runs:/{run_id}/artifacts/model",
            "training-model-psystock",
        )


if __name__ == "__main__":
    RUN_ID = sys.argv[1]
    register_model(run_id=RUN_ID)
