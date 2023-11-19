"""
Prepares raw data and provides training, validation and test datasets
"""

import argparse
from pathlib import Path
import os
import numpy as np
import pandas as pd
import mlflow

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

# Read in data
data = pd.read_csv('data/taxi-data.csv')  
data = data[NUMERIC_COLS + CAT_NOM_COLS + CAT_ORD_COLS + [TARGET_COL]]

# Split data
random_data = np.random.rand(len(data))
msk_train = random_data < 0.8
msk_test = random_data >= 0.8
train = data[msk_train]
test = data[msk_test]

# export data as parquet
train.to_parquet("output/train.parquet")
test.to_parquet("output/test.parquet")

# SOME TODO ITEMS
# Can we log some metadata?
# Can we avoid hardcoding paths?
# Can we wrap some of this code into functions and guard them better?