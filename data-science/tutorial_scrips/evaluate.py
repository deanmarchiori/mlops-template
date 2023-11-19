# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Evaluates trained ML model using test dataset.
Saves predictions, evaluation results
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os
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

# load model 
model =  mlflow.sklearn.load_model("output/model") 

# Load the test data
test_data = pd.read_parquet("output/test.parquet")

y_test = test_data[TARGET_COL]
X_test = test_data[NUMERIC_COLS + CAT_NOM_COLS + CAT_ORD_COLS]

# Get predictions to y_test (y_test)
yhat_test = model.predict(X_test)

# Save the output data with feature columns, predicted cost, and actual cost in csv file
output_data = X_test.copy()
output_data["real_label"] = y_test
output_data["predicted_label"] = yhat_test
output_data.to_csv( "output/predictions.csv")

# Evaluate Model performance with the test set
r2 = r2_score(y_test, yhat_test)
mse = mean_squared_error(y_test, yhat_test)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, yhat_test)

# Visualize results
plt.scatter(y_test, yhat_test,  color='black')
plt.plot(test_data[TARGET_COL].values, test_data[TARGET_COL].values, color='blue', linewidth=3)
plt.xlabel("Real value")
plt.ylabel("Predicted value")
plt.title("Comparing Model Predictions to Real values - Test Data")
plt.savefig("output/predictions.png")

