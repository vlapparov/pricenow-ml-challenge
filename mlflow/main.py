import click
import os


import mlflow
from mlflow.tracking import MlflowClient
from mlflow.utils import mlflow_tags
from mlflow.entities import RunStatus
from mlflow.utils.logging_utils import eprint

from mlflow.tracking.fluent import _get_experiment_id


def _already_ran(entry_point_name, parameters, git_commit, experiment_id=None):
    """Best-effort detection of if a run with the given entrypoint name,
    parameters, and experiment id already ran. The run must have completed
    successfully and have at least the parameters provided.
    """
    experiment_id = experiment_id if experiment_id is not None else _get_experiment_id()
    client = MlflowClient()
    all_runs = reversed(client.search_runs([experiment_id]))
    for run in all_runs:
        tags = run.data.tags
        if tags.get(mlflow_tags.MLFLOW_PROJECT_ENTRY_POINT, None) != entry_point_name:
            continue
        match_failed = False
        for param_key, param_value in parameters.items():
            run_value = run.data.params.get(param_key)
            if run_value != param_value:
                match_failed = True
                break
        if match_failed:
            continue

        if run.info.to_proto().status != RunStatus.FINISHED:
            eprint(
                ("Run matched, but is not FINISHED, so skipping (run_id=%s, status=%s)")
                % (run.info.run_id, run.info.status)
            )
            continue

        previous_version = tags.get(mlflow_tags.MLFLOW_GIT_COMMIT, None)
        if git_commit != previous_version:
            eprint(
                (
                    "Run matched, but has a different source version, so skipping "
                    "(found=%s, expected=%s)"
                )
                % (previous_version, git_commit)
            )
            continue
        return client.get_run(run.info.run_id)
    eprint("No matching run has been found.")
    return None


def _get_or_run(entrypoint, parameters, git_commit, use_cache=True):
    existing_run = _already_ran(entrypoint, parameters, git_commit)
    if use_cache and existing_run:
        print(
            "Found existing run for entrypoint={} and parameters={}".format(
                entrypoint, parameters
            )
        )
        return existing_run
    print(
        "Launching new run for entrypoint={} and parameters={}".format(
            entrypoint, parameters
        )
    )
    submitted_run = mlflow.run(
        ".", entrypoint, parameters=parameters, env_manager="local"
    )
    return MlflowClient().get_run(submitted_run.run_id)


@click.command()
@click.option("--max-row-limit", default=10000, type=int)
@click.option(
    "--features",
    default="releaseyear,key,acousticness,danceability,energy,instrumentalness,liveness,loudness,speechiness,valence,tempo",
    help="Comma-separated list of features to use in the model",
)
@click.option(
    "--target-feature",
    default="score",
    help="Target variable to use in the model",
)
@click.option("--n-estimators", default=100, type=int)
@click.option("--learning-rate", default=0.1, type=float)
def workflow(max_row_limit, features, target_feature, n_estimators, learning_rate):
    # Note: The entrypoint names are defined in MLproject. The artifact directories
    # are documented by each step's .py file.
    with mlflow.start_run() as active_run:
        os.environ["SPARK_CONF_DIR"] = os.path.abspath(".")
        git_commit = active_run.data.tags.get(mlflow_tags.MLFLOW_GIT_COMMIT)
        load_raw_data_run = _get_or_run("load_raw_data", {}, git_commit)
        music_csv_uri = os.path.join(
            load_raw_data_run.info.artifact_uri, "music-csv-dir"
        )
        etl_data_run = _get_or_run(
            "etl_data",
            {
                "music_csv": music_csv_uri,
                "features": features,
                "target_feature": target_feature,
                "max_row_limit": max_row_limit,
            },
            git_commit,
        )
        processed_data_uri = os.path.join(
            etl_data_run.info.artifact_uri, "music-processed-dir"
        )

        gbm_params = {
            "processed_data": processed_data_uri,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
        }
        _get_or_run("train", gbm_params, git_commit, use_cache=False)


if __name__ == "__main__":
    workflow()
