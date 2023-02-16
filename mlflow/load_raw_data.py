import requests
import tempfile
import os
import mlflow
import click


@click.command(
    help="Downloads the dataset and saves it as an mlflow artifact called 'music-csv-dir'."
)
@click.option(
    "--url",
    default="https://raw.githubusercontent.com/vlapparov/pricenow-ml-challenge/dev/data/pitchfork.csv",
)
def load_raw_data(url):
    with mlflow.start_run() as mlrun:
        local_dir = tempfile.mkdtemp()
        local_filename = os.path.join(local_dir, "pitchfork.csv")
        print("Downloading {} to {}".format(url, local_filename))
        r = requests.get(url, stream=True)
        with open(local_filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

        print("Uploading data: %s" % local_filename)
        mlflow.log_artifact(local_filename, "music-csv-dir")


if __name__ == "__main__":
    load_raw_data()
