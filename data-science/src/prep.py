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

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("prep")
    parser.add_argument("--raw_data", type=str, help="Path to raw data")
    parser.add_argument("--train_data", type=str, help="Path to train dataset")
    parser.add_argument("--test_data", type=str, help="Path to test dataset")

    args = parser.parse_args()

    return args


def main(args):

    # Read in data
    data = pd.read_csv((Path(args.raw_data)))
    data = data[NUMERIC_COLS + CAT_NOM_COLS + CAT_ORD_COLS + [TARGET_COL]]

    # Split data
    random_data = np.random.rand(len(data))
    msk_train = random_data < 0.8
    msk_test = random_data >= 0.8
    train = data[msk_train]
    test = data[msk_test]

    # Enable logging using MLFlow
    mlflow.log_metric('train size', train.shape[0])
    mlflow.log_metric('test size', test.shape[0])

    # export data as parquet
    train.to_parquet((Path(args.train_data) / "train.parquet"))
    test.to_parquet((Path(args.test_data) / "test.parquet"))



if __name__ == "__main__":

    mlflow.start_run()

    args = parse_args()
    main(args)
    
    mlflow.end_run()

    