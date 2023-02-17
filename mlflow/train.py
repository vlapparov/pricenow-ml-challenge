import os
import warnings
import click

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


@click.command(help="Trains a Gradient Boosting model")
@click.option("--processed-data", help="Path to processed data file", type=str)
@click.option("--n-estimators", default=100, type=int)
@click.option("--learning-rate", default=0.1, type=float)
def train_gbm(processed_data, n_estimators, learning_rate):
    warnings.filterwarnings("ignore")
    file_path = os.path.join(processed_data, "processed.csv")
    data = pd.read_csv(file_path)

    X = data.copy()
    y = X.pop("score")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=123
    )

    with mlflow.start_run():
        gb_regressor = GradientBoostingRegressor(
            n_estimators=n_estimators, learning_rate=learning_rate
        )
        gb_regressor.fit(X_train, y_train)
        y_pred = gb_regressor.predict(X_test)

        (rmse, mae, r2) = eval_metrics(y_test, y_pred)

        print(
            "GradientBoostingRegressor model (n_estimators={:f}, learning_rate={:f}):".format(
                n_estimators, learning_rate
            )
        )
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                gb_regressor, "model", registered_model_name="GradientBoostingRegressor"
            )
        else:
            mlflow.sklearn.log_model(gb_regressor, "model")


if __name__ == "__main__":
    train_gbm()
