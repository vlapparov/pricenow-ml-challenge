# Music Wizard ML Workflow 

This simple ML workflow is an example of how to
chain together multiple different MLflow runs which each encapsulate
a transformation or training step, allowing a clear definition of the
interface between the steps, as well as allowing for caching and reuse
of the intermediate results.

At a high level, our goal is to predict whether an album will be well received or not by predicting users score. 

This example is based
on [this tutorial](https://github.com/mlflow/mlflow/tree/master/examples/multistep_workflow).

## The Workflow

![image](../assets/flow_diagram.png)

There are three steps to this workflow:

- **load_raw_data.py**: downloads the music album dataset with reviews as a CSV and puts it into the artifact store. For this excercise, we will use Github as our data store.

- **etl_data.py**: performs data preprocessing step along with feature engineering and puts the results into the artifact store

- **train.py**: Trains a GradientBoostingRegressor model using the preprocessed data and puts the results into the artifact store

While we can run each of these steps manually, here we have a driver
run, defined as **main** (main.py). This run will run
the steps in order, passing the results of one to the next.
Additionally, this run will attempt to determine if a sub-run has
already been executed successfully with the same parameters and, if so,
reuse the cached results.

## Running this Example
In order for the multistep workflow to find the other steps, you must
execute ``mlflow run`` from this directory. 

Simply run:

```bash
cd mlflow
mlflow run . --env-manager local
```

By default, wherever you run your program, the tracking API writes data into files into a local **./mlruns** directory. 

You can then run MLflow’s Tracking UI:
```bash
mlflow ui
```

and view it at http://localhost:5000

To modify parameters used in the workflow you can modify the **MLproject** file or pass model parameters 
to the run command:

```bash
mlflow run . -P n_estimators=400 -P learning_rate=0.05
```

The list of parameters that you could configure:
- **max_row_limit**: limits the data size to run comfortably on a laptop
- **n_estimators**: The number of boosting stages to perform. Values must be in the range `[1, inf)`
- **learning_rate**: shrinks the contribution of each tree by `learning_rate`.  Values must be in the range `[0.0, inf)`
- **features**: comma-separated list of features to use in the model
- **target_feature**: target variable to use in the model

## Model Serving

In order to serve the model for a certain run, we can use the 
```bash
mlflow models serve -m runs:/<RUN_ID>/model
```

