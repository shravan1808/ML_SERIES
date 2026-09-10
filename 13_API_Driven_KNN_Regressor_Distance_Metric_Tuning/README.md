PROJECT 3 — API-DRIVEN KNN REGRESSOR & DISTANCE METRIC HYPERPARAMETER TUNING
Difficulty: Hard

Platform: Google Colab

Language: Python

Libraries: Requests, Pandas, NumPy, Scikit-Learn

Submission: Google Colab Notebook

Problem Statement:
-------------------
An urban transportation authority wants to predict real-time ride-hailing trip durations (Duration_Minutes) based on spatial and temporal telemetry. Instead of relying on pre-loaded local files, students must build a custom REST API data ingestion pipeline using requests to dynamically fetch live JSON payloads from remote endpoints, parse them into DataFrames, merge 3 relational entities (Trip Logs, Route Geography, and Vehicle Specs) across 50 records, implement a K-Nearest Neighbors (KNN) Regressor, tune distance metrics (Euclidean vs. Manhattan), and evaluate error metrics across varying values of K.


REST API Endpoints Provided
-----------------------------
Use these active endpoints inside your fetching function:


TRIPS_API_ENDPOINT = "https://raw.githubusercontent.com/datasets-repository/transport-api/main/trip_logs.json"
ROUTES_API_ENDPOINT = "https://raw.githubusercontent.com/datasets-repository/transport-api/main/route_geography.json"
VEHICLES_API_ENDPOINT = "https://raw.githubusercontent.com/datasets-repository/transport-api/main/vehicle_specs.json"


YOUR IMPLEMENTATION SPECIFICATION
------------------------------------------
Requirement: Implement the fetch_api_data(url: str) function using requests.get(). 
Your implementation must raise status errors (raise_for_status()), handle network exceptions gracefully, parse the returned JSON into a pandas.DataFrame, and retrieve all 3 remote datasets dynamically.


# =====================================================================
# STUDENT TASK: IMPLEMENT API FETCHING LOGIC BELOW
# =====================================================================
import requests
import pandas as pd
import numpy as np

def fetch_api_data(url: str) -> pd.DataFrame:
    """
    Student Implementation Required:
    1. Send an HTTP GET request to the provided URL using requests.
    2. Check for HTTP status code errors.
    3. Convert the returned JSON response payload into a Pandas DataFrame.
    4. Implement error handling (try-except block) for network failures.
    """
    # Write your code here
    pass

# Retrieve datasets via your API function
df_trips = fetch_api_data(TRIPS_API_ENDPOINT)
df_routes = fetch_api_data(ROUTES_API_ENDPOINT)
df_vehicles = fetch_api_data(VEHICLES_API_ENDPOINT)



TASKS TO IMPLEMENT
-------------------

TASK 1 — Custom REST API Ingestion & Multi-Table Spatial-Temporal Merging
Write the fetch_api_data() pipeline to programmatically pull records via HTTP REST requests. Confirm successful payload parsing (HTTP 200 OK) and perform sequential inner joins on df_trips, df_routes, and df_vehicles using primary keys (Route_ID, Vehicle_ID).

Expected Output:
HTTP REST Status      : 200 OK (3 API Endpoints Synced Successfully)
Merged Dataset Shape : (50, 8)
Sample Records:
  Trip_ID Route_ID Vehicle_ID  Pickup_Hour  ...  Congestion_Index  Engine_Size_L  Duration_Minutes
0  TRP_800    RTE_1      VEH_1            8  ...               2.1            1.6              12.5
1  TRP_801    RTE_2      VEH_2           14  ...               6.8            2.5              28.0
2  TRP_802    RTE_3      VEH_3            2  ...               1.2            1.4               8.2
3  TRP_803    RTE_4      VEH_4           18  ...               8.5            3.0              45.0
4  TRP_804    RTE_5      VEH_5            9  ...               4.2            2.0              18.5


TASK 2 — Mandatory Feature Standardization
Isolate continuous features (Distance_KM, Congestion_Index, Pickup_Hour, Engine_Size_L) and target (Duration_Minutes). Apply StandardScaler.

Expected Output:
Continuous Feature Set   : Distance_KM, Congestion_Index, Pickup_Hour, Engine_Size_L
Scaled Features Check    : Mean = 0.00, Variance = 1.00
Target (y) Span          : Min = 6.2 Mins, Max = 46.5 Mins, Mean = 22.8 Mins


TASK 3 — Stratified Splitting for Regression Bins
Binner continuous target y into 3 discrete quantile bins to perform a Stratified Train-Test Split (80/20 ratio), ensuring representative target distributions.

Expected Output:
Train-Test Dimensions:
- X_train Shape : (40, 4)
- X_test Shape  : (10, 4)
- y_train Mean  : 22.84 Mins
- y_test Mean   : 22.66 Mins


TASK 4 — Hyperparameter Tuning across K Values & Metrics
Evaluate KNeighborsRegressor across K = [1, 3, 5, 7] using both Euclidean (p=2) and Manhattan (p=1) distance metrics. Compute Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) for each configuration.

Expected Output:
Hyperparameter Tuning Results:
- K=1 | Metric=Euclidean : MAE = 0.68 Mins | RMSE = 0.85 Mins
- K=3 | Metric=Euclidean : MAE = 0.52 Mins | RMSE = 0.64 Mins  <-- (OPTIMAL)
- K=5 | Metric=Euclidean : MAE = 0.74 Mins | RMSE = 0.92 Mins
- K=7 | Metric=Euclidean : MAE = 1.15 Mins | RMSE = 1.38 Mins
- K=3 | Metric=Manhattan : MAE = 0.58 Mins | RMSE = 0.71 Mins


TASK 5 — Prediction vs Actual Residual Analysis
Generate final test set predictions using the optimal configuration (K=3, Euclidean). Compute residual errors (Actual - Predicted).

Expected Output:
Test Sample Predictions (K=3, Euclidean):
Sample 1 : Actual = 12.50 Mins | Predicted = 12.43 Mins | Residual = +0.07 Mins
Sample 2 : Actual = 28.00 Mins | Predicted = 28.10 Mins | Residual = -0.10 Mins
Sample 3 : Actual =  8.20 Mins | Predicted =  8.07 Mins | Residual = +0.13 Mins
Sample 4 : Actual = 45.00 Mins | Predicted = 45.10 Mins | Residual = -0.10 Mins
Sample 5 : Actual = 18.50 Mins | Predicted = 18.47 Mins | Residual = +0.03 Mins


FINAL EXPECTED OUTPUT
----------------------
========== API-DRIVEN KNN REGRESSOR & HYPERPARAMETER TUNING ==========

Data Ingestion Status      : REST API Ingestion Successful (HTTP 200)
Master Telemetry Records   : 50
Feature Inputs             : Distance_KM, Congestion_Index, Pickup_Hour, Engine_Size_L
Target Variable            : Duration_Minutes (Continuous Regression)

Optimal Model Configuration:
- Optimal K Neighbors      : K = 3
- Distance Metric          : Euclidean Distance (p = 2)
- Feature Preprocessing    : StandardScaler Applied

Model Evaluation Summary:
- Test Set Mean Absolute Error (MAE) : 0.52 Minutes
- Test Set Root Mean Squared Error   : 0.64 Minutes
- Mean Prediction Residual           : +/- 0.08 Minutes

Conclusion:
Students must successfully construct dynamic REST API consumers to decouple data acquisition from machine learning workflows. KNN Regression models local spatial-temporal trip durations accurately when distance metrics are calculated on standardized features. Hyperparameter evaluation confirms K=3 balances local neighborhood sensitivity without overfitting to single-sample noise.