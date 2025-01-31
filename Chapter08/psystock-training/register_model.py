import sys

import mlflow
from mlflow import MlflowClient


def register_model(run_id: str):
    with mlflow.start_run(run_name="register_model") as run:
        mlflow.set_tag("mlflow.runName", "register_model")

        register_name = "training-model-psystock"

        result = mlflow.register_model(
            f"runs:/{run_id}/artifacts/model",
            register_name,
        )
        print(f"{result = }")
        print(f"Version: {result.version}")

        client = MlflowClient()
        client.set_registered_model_alias(register_name, "current", result.version)


if __name__ == "__main__":
    RUN_ID = sys.argv[1]
    register_model(run_id=RUN_ID)
