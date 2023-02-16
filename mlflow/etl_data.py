import tempfile
import os
import mlflow
import click
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing


def remove_duplicates(df):
    data = df.copy()
    return data.drop_duplicates(subset=["artist", "album"]).reset_index(drop=True)


def preprocess_genre_column(df):
    output = df.copy()
    output.loc[
        (output["genre"].isna()) | (output["genre"] == "none"),
        "genre",
    ] = "Other"
    return output


def build_previous_avg_features(df, selected_features):
    # calculate average of all features except for the last row
    assert all(col in df.columns for col in selected_features)
    return pd.DataFrame(df[selected_features].iloc[:-1].mean()).T.add_suffix(
        "_previous"
    )


def create_features_per_group(group, features_selected_for_previous_avg):
    # calculate average of all features except for the last row
    previous_avg_features = build_previous_avg_features(
        group, selected_features=features_selected_for_previous_avg
    )
    # get the last record within the group
    last_record = (
        group[features_selected_for_previous_avg].iloc[-1:].reset_index(drop=True)
    )
    current_merged = pd.concat([last_record, previous_avg_features], axis=1)
    return current_merged


def get_one_hot_genres(df):
    assert (
        "genre" in df.columns
    ), "Provided DataFrame doesn't contain 'genre' among its columns"
    data = df.copy()
    return pd.get_dummies(data["genre"]).add_suffix("_onehot")


@click.command(
    help="Given a CSV file (see load_raw_data), preprocesses it and saves it as an mlflow artifact"
)
@click.option("--music-csv")
@click.option(
    "--max-row-limit",
    default=10000,
    help="Limit the data size to run comfortably on a laptop.",
)
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
def etl_data(music_csv, max_row_limit, features, target_feature):
    with mlflow.start_run() as mlrun:
        tmpdir = tempfile.mkdtemp()
        processed_data_dir = os.path.join(tmpdir, "processed_data")
        print("Preprocessing CSV {} ".format(music_csv))
        file_path = os.path.join(music_csv, "pitchfork.csv")
        df = pd.read_csv(file_path)
        mlflow.log_param("target_feature", target_feature)
        if target_feature not in df.columns:
            ex = ValueError(
                "Target variable '{}' not found in CSV".format(target_feature)
            )
            mlflow.set_tag("exception", str(ex))
            raise ex
        feature_list = features.split(",")
        features_not_found = []
        music_features = []
        mlflow.log_param("input_features", features)
        for feature in feature_list:
            if feature not in df.columns:
                features_not_found.append(feature)
            else:
                music_features.append(feature)
        mlflow.log_param("features_not_found", ",".join(features_not_found))
        mlflow.log_param("music_features", ",".join(music_features))
        df = remove_duplicates(df)
        print("Count the number of albums per artist")
        album_counts = df["artist"].value_counts()
        print("Get a list of artists with more than one album")
        artists_with_multiple_albums = album_counts[album_counts > 1].index.tolist()
        mlflow.log_metric(
            "artists_with_multiple_albums", len(artists_with_multiple_albums)
        )
        print("Create a new DataFrame with one row per artist")
        new_df = (
            df[df["artist"].isin(artists_with_multiple_albums)]
            .sort_values(by=["artist", "releaseyear"])
            .reset_index(drop=True)
        )
        print("Preprocess data and create one-hot variables")
        new_df = preprocess_genre_column(new_df)
        onehot_genres = get_one_hot_genres(new_df)
        new_df_one_hot = pd.concat([new_df, onehot_genres], axis=1)
        one_hot_columns = list(onehot_genres.columns)
        features_selected_for_previous_avg = (
            music_features + one_hot_columns + [target_feature]
        )
        mlflow.log_param(
            "features_selected_for_previous_avg",
            ",".join(features_selected_for_previous_avg),
        )
        # create new DataFrame that would be used for GBM training
        print("Preparing feature set...")
        feature_chunks = Parallel(n_jobs=multiprocessing.cpu_count())(
            delayed(create_features_per_group)(
                group, features_selected_for_previous_avg
            )
            for _, group in new_df_one_hot.head(max_row_limit).groupby("artist")
        )
        processed = pd.concat(feature_chunks)
        mlflow.log_metric("data_rows", processed.shape[0])
        mlflow.log_metric("data_cols", processed.shape[1])
        if not os.path.exists(processed_data_dir):
            os.makedirs(processed_data_dir)
            print("Created directory: %s" % processed_data_dir)
        processed.to_csv(os.path.join(processed_data_dir, "processed.csv"), index=False)
        print("Uploading processed data to path: %s" % processed_data_dir)
        mlflow.log_artifacts(processed_data_dir, "music-processed-dir")


if __name__ == "__main__":
    etl_data()
