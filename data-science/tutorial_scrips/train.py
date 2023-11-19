"""
Trains ML model using training dataset. Saves trained model.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import mlflow
import mlflow.sklearn

TARGET_COL = "cost"

NUMERIC_COLS = [
    "distance",
    "dropoff_latitude",
    "dropoff_longitude",
    "passengers",
    "pickup_latitude",
    "pickup_longitude",
    "pickup_weekday",
    "pickup_month",
    "pickup_monthday",
    "pickup_hour",
    "pickup_minute",
    "pickup_second",
    "dropoff_weekday",
    "dropoff_month",
    "dropoff_monthday",
    "dropoff_hour",
    "dropoff_minute",
    "dropoff_second",
]

CAT_NOM_COLS = [
    "store_forward",
    "vendor",
]

CAT_ORD_COLS = [
]

# Read train data
train_data = pd.read_parquet('output/train.parquet')

# Split the data into input(X) and output(y)
Y_train = train_data[TARGET_COL]
X_train = train_data[NUMERIC_COLS + CAT_NOM_COLS + CAT_ORD_COLS]

# Train a Random Forest Regression Model with the training set
model = RandomForestRegressor(n_estimators = 500,
                    bootstrap = 1,
                    max_depth = 10,
                    max_features = 1.0,
                    min_samples_leaf = 4,
                    min_samples_split = 5,
                    random_state=0)

# Train model with the train set
model.fit(X_train, Y_train)

mlflow.sklearn.save_model(sk_model=model, path="output/model")

