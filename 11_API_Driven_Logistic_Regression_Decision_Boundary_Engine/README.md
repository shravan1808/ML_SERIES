WEEK 4 — DAY 19
-----------------
PROJECT 11 — API-DRIVEN LOGISTIC REGRESSION & DECISION BOUNDARY ENGINE

Platform: Google Colab

Language: Python

Libraries: Requests, Pandas, NumPy, Scikit-Learn

Submission: Google Colab Notebook

Problem Statement
--------------------
An enterprise cybersecurity firm builds automated threat detection engines to classify incoming network requests as either Normal (0) or Malicious (1). Instead of relying on pre-loaded local data dictionary definitions, students must build a custom REST API data ingestion pipeline using requests to dynamically fetch live JSON payloads from 3 remote endpoints (Network Flow Logs, Host Telemetry, and Threat Intelligence Metadata). You will then join these datasets across 50 records, train a Logistic Regression model, extract sigmoid probability thresholds, and quantify how classification metrics change across varying decision boundaries.  

REST API Endpoints Provided
-----------------------------
Use these active endpoints inside your fetching function:
FLOW_LOGS_API_ENDPOINT = "https://raw.githubusercontent.com/datasets-repository/security-api/main/flow_logs.json"
HOST_TELEMETRY_API_ENDPOINT = "https://raw.githubusercontent.com/datasets-repository/security-api/main/host_telemetry.json"
THREAT_INTEL_API_ENDPOINT = "https://raw.githubusercontent.com/datasets-repository/security-api/main/threat_intel.json"



STUDENT IMPLEMENTATION SPECIFICATION
---------------------------------------
Requirement: Implement the fetch_api_data(url: str) function using requests.get(). Your implementation must handle HTTP errors (raise_for_status()), manage network exceptions gracefully, parse the returned JSON payload into a pandas.DataFrame, and dynamically retrieve all 3 datasets.


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
    2. Check for HTTP status code errors using response.raise_for_status().
    3. Convert the returned JSON response payload into a Pandas DataFrame.
    4. Implement error handling (try-except block) for network failures.
    """
    # Write your code here
    pass

# Retrieve datasets via your API function
df_flow = fetch_api_data(FLOW_LOGS_API_ENDPOINT)
df_host = fetch_api_data(HOST_TELEMETRY_API_ENDPOINT)
df_intel = fetch_api_data(THREAT_INTEL_API_ENDPOINT)




TASKS TO IMPLEMENT
------------------
TASK 1 — Enterprise Multi-Table REST Data Fusion
Write the fetch_api_data() function to pull payloads via HTTP REST requests. Perform sequential inner joins on df_flow, df_host, and df_intel using primary/foreign keys (Host_ID, Threat_Intel_ID).

Expected Output:
HTTP REST Status      : 200 OK (3 Security API Endpoints Synced)
Merged Dataset Shape : (50, 9)
Sample Output:
  Flow_ID Host_ID Threat_Intel_ID  Packet_Size_Bytes  ...  CPU_Utilization_Pct  IP_Reputation_Score  Is_Malicious
0  FLW_100   HST_1          INT_1                150  ...                 12.5                 0.05             0
1  FLW_101   HST_2          INT_2               4200  ...                 88.4                 0.88             1
2  FLW_102   HST_3          INT_3                210  ...                 15.0                 0.12             0
3  FLW_103   HST_4          INT_4               5800  ...                 92.1                 0.95             1
4  FLW_104   HST_5          INT_5                180  ...                 10.8                 0.08             0


TASK 2 — Feature Scaling & Stratified Train-Test Split
Extract features (X) excluding identifiers (Flow_ID, Host_ID, Threat_Intel_ID) and target (Is_Malicious). Standardize feature scales using StandardScaler. Perform a 70/30 Stratified Split.

Expected Output:
Features Matrix (X) Shape : (50, 6)
Scaled Feature Set       : Mean = 0.00, Std = 1.00
Train-Test Split Shapes  : X_train = (35, 6), X_test = (15, 6)
Target Class Ratio       : Train = 51.4% (18/17) | Test = 46.7% (7/8)


TASK 3 — Logistic Regression Model Fitting & Parameter Inspection
Fit a LogisticRegression model on training data. Extract feature coefficients and model intercept to inspect parameter weights.

Expected Output:
Model Parameters:
- Intercept Weight (Beta_0) : -0.1542
- Feature Coefficients (Beta_i):
  - Packet_Size_Bytes       : +1.2840
  - Connection_Duration_Sec : +1.1025
  - CPU_Utilization_Pct     : +1.4512
  - Failed_Login_Attempts   : +1.3210
  - IP_Reputation_Score     : +1.5890
  - Known_Vulnerability_Flag: +0.9420


TASK 4 — Raw Probability Extraction & Sigmoid Output Audit
Generate raw predicted probabilities using predict_proba() on X_test. Display the decision confidence scores for the first 5 test samples.

Expected Output:
Test Sample Sigmoid Probabilities (P(y=1|X)):
Sample 1 (True Class 0) : P(Malicious) = 0.0215 -> Low Risk
Sample 2 (True Class 1) : P(Malicious) = 0.9842 -> Critical Risk
Sample 3 (True Class 0) : P(Malicious) = 0.0410 -> Low Risk
Sample 4 (True Class 1) : P(Malicious) = 0.9915 -> Critical Risk
Sample 5 (True Class 0) : P(Malicious) = 0.0180 -> Low Risk


TASK 5 — Custom Threshold Sensitivity Analysis
Evaluate model accuracy and predictions on X_test across three distinct probability decision boundaries: Strict (Threshold = 0.30), Standard (Threshold = 0.50), and Conservative (Threshold = 0.70).

Expected Output:
Threshold Evaluation:
- Threshold = 0.30 : Accuracy = 100.0% | Predicted Positive Count = 7
- Threshold = 0.50 : Accuracy = 100.0% | Predicted Positive Count = 7
- Threshold = 0.70 : Accuracy = 100.0% | Predicted Positive Count = 7


FINAL EXPECTED OUTPUT
----------------------
========== API-DRIVEN LOGISTIC REGRESSION & DECISION BOUNDARY ENGINE ==========

Data Ingestion Status      : REST API Ingestion Successful (HTTP 200)
Master Dataset Records     : 50
Features Included          : 6 (Continuous & Binary Indicators)
Target Output              : Is_Malicious (Binary Classification)

Model Training Metrics:
- Stratified Train Split   : 35 Records
- Stratified Test Split    : 15 Records
- Base Test Accuracy       : 100.0% (at Default 0.50 Threshold)

Sigmoid Probability Profiling:
- Maximum Malicious Prob   : 99.15%
- Minimum Malicious Prob   : 1.80%
- Key Probability Drivers  : IP_Reputation_Score (+1.5890), CPU_Utilization_Pct (+1.4512)

Decision Threshold Tuning:
- Threshold @ 0.30 (Strict): 100.0% Accuracy (Maximizes recall for critical threats)
- Threshold @ 0.50 (Normal): 100.0% Accuracy (Balanced baseline)
- Threshold @ 0.70 (Alert) : 100.0% Accuracy (Minimizes security alert fatigue)

Conclusion:
By implementing dynamic REST API fetching, the data pipeline dynamically extracts live security metrics. Logistic Regression maps continuous multi-source features to calibrated probability scores, enabling security teams to set custom operational thresholds based on risk tolerance.