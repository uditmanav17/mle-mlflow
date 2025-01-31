import os

import click
import mlflow

os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
print(os.environ.get("MLFLOW_TRACKING_URI"))


def _run(entrypoint, parameters=None):
    if parameters is None:
        parameters = {}
    print(f"Launching new run for entrypoint={entrypoint} and parameters={parameters}")
    submitted_run = mlflow.run(".", entrypoint, parameters=parameters)
    return mlflow.tracking.MlflowClient().get_run(submitted_run.run_id)


@click.command()
def workflow():
    with mlflow.start_run(run_name="pystock-training") as active_run:
        mlflow.set_tag("mlflow.runName", "pystock-training")
        train_run = _run("train_model")
        train_run_id = train_run.info.run_id
        print(f"Train run ID - {train_run_id}")
        evaluate_run = _run("evaluate_model")
        eval_run_id = evaluate_run.info.run_id
        print(f"Evaluate run ID - {eval_run_id}")

        model_uri = os.path.join(train_run.info.artifact_uri, "model")
        mlflow.register_model(model_uri, "training-model-psystock")
        print(f"{model_uri = }")

        # train_run_id = "0a13af4b8b8d4fbd8a4d09d148cb8429"
        _run("register_model", {"run_id": train_run_id})


if __name__ == "__main__":
    workflow()
