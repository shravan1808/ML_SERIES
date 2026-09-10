PROJECT 12 — API-DRIVEN DECISION TREE CLASSIFIER & FEATURE IMPORTANCE ANALYTICS
-------------
Platform: Google Colab

Language: Python

Libraries: Requests, Pandas, Scikit-Learn

Submission: Google Colab Notebook

Problem Statement
-------------------
A retail banking corporation wants to automate personal loan approval decisions (Approved = 1, Rejected = 0) using transparent, rule-based decision trees. Instead of hardcoding static applicant dataframes, students must build an API ingestion pipeline using requests to dynamically retrieve raw credit and financial logs from REST endpoints. You must combine 3 relational datasets (Applicant Profiles, Credit Bureau Metrics, and Employment Logs) across 50 records, train a Decision Tree Classifier, extract Gini impurity feature importance scores, and evaluate structural pruning to prevent model overfitting.  


REST API Endpoints Provided
--------------------------
Use these active endpoints inside your fetching function:

APPLICANTS_API_ENDPOINT = "https://raw.githubusercontent.com/datasets-repository/banking-api/main/applicant_profiles.json"
BUREAU_API_ENDPOINT = "https://raw.githubusercontent.com/datasets-repository/banking-api/main/credit_bureau.json"
EMPLOYMENT_API_ENDPOINT = "https://raw.githubusercontent.com/datasets-repository/banking-api/main/employment_logs.json"


YOUYR IMPLEMENTATION SPECIFICATION
------------------------------------
Requirement: Implement the fetch_api_data(url: str) function using requests.get(). Your implementation must raise status errors (raise_for_status()), handle network exceptions gracefully, parse the returned JSON into a pandas.DataFrame, and retrieve all 3 banking datasets dynamically.


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
df_app = fetch_api_data(APPLICANTS_API_ENDPOINT)
df_bureau = fetch_api_data(BUREAU_API_ENDPOINT)
df_emp = fetch_api_data(EMPLOYMENT_API_ENDPOINT)


TASKS TO IMPLEMENT
-------------------
TASK 1 — REST API Ingestion & Multi-Source Relational Fusion
Implement fetch_api_data() to dynamically fetch financial logs. Join df_app, df_bureau, and df_emp on primary keys (Bureau_ID, Emp_ID). Apply One-Hot Encoding to categorical feature (Loan_Intent).

Expected Output:
HTTP REST Status      : 200 OK (3 Banking API Endpoints Synced)
Merged Dataset Shape : (50, 9)
Sample Output:
  Applicant_ID Bureau_ID Emp_ID  Requested_Loan_USD  ...  Debt_To_Income_Ratio  Annual_Income_USD  Is_Approved
0      APP_300     BUR_1  EMP_1               10000  ...                  0.18              85000            1
1      APP_301     BUR_2  EMP_2               45000  ...                  0.45              32000            0
2      APP_302     BUR_3  EMP_3                8000  ...                  0.12              98000            1
3      APP_303     BUR_4  EMP_4               60000  ...                  0.52              41000            0
4      APP_304     BUR_5  EMP_5               12000  ...                  0.22              79000            1


TASK 2 — Stratified Splitting & Baseline Tree Fitting
Separate feature matrix (X) and target vector (y = Is_Approved). Perform an 80/20 Stratified Train-Test Split. Train an unpruned DecisionTreeClassifier(criterion='gini', random_state=42).

Expected Output:
Train-Test Shapes       : X_train = (40, 7), X_test = (10, 7)
Unpruned Tree Metrics   :
- Training Accuracy     : 100.0%
- Testing Accuracy      : 100.0%
- Tree Maximum Depth    : 2
- Total Leaf Nodes      : 3


TASK 3 — Gini Feature Importance Extraction
Extract feature_importances_ from the fitted tree model. Format as a ranked summary table.

Expected Output:
Gini Feature Importance Rankings:
1. Credit_Score          : 0.8245 (Primary Root Split Attribute)
2. Debt_To_Income_Ratio  : 0.1755 (Secondary Node Split Attribute)
3. Requested_Loan_USD    : 0.0000
4. Annual_Income_USD     : 0.0000
5. Employment_Years      : 0.0000
6. Loan_Intent_Home      : 0.0000
7. Loan_Intent_Personal  : 0.0000


TASK 4 — Decision Rules Extraction
Export human-readable text decision rules using export_text() from Scikit-Learn.

Expected Output:
Extracted Decision Rules:
|--- Credit_Score <= 665.00
|   |--- class: 0 (Loan Rejected)
|--- Credit_Score >  665.00
|   |--- Debt_To_Income_Ratio <= 0.33.50
|   |   |--- class: 1 (Loan Approved)
|   |--- Debt_To_Income_Ratio >  0.33.50
|   |   |--- class: 0 (Loan Rejected)


TASK 5 — Hyperparameter Pruning & Overfitting Mitigation
Train a pruned tree using max_depth=2 and min_samples_leaf=5. Compare training vs testing performance against unpruned baseline.

Expected Output:
-----------------------
Pruned Tree Evaluation (max_depth=2, min_samples_leaf=5):
- Pruned Tree Train Accuracy : 100.0%
- Pruned Tree Test Accuracy  : 100.0%
- Tree Complexity Reduction  : Depth maintained at 2, leaf nodes restricted from over-expanding.



========== API-DRIVEN DECISION TREE CLASSIFIER & FEATURE IMPORTANCE ANALYTICS ==========

Data Ingestion Status            : REST API Ingestion Successful (HTTP 200)
Master Applicant Dataset Records : 50
Feature Space Dimension          : 7 Predictive Attributes
Target Output                    : Is_Approved (Binary Decision)

Decision Tree Performance:
- Training Accuracy (Unpruned)   : 100.0%
- Testing Accuracy (Unpruned)    : 100.0%
- Pruned Tree Test Accuracy      : 100.0% (max_depth=2)

Top Gini Feature Importances:
1. Credit_Score                  : 82.45% Contribution (Primary Decision Splitting Threshold)
2. Debt_To_Income_Ratio          : 17.55% Contribution (Secondary Risk Discriminator)

Key Decision Rule Path:
Applicants with Credit_Score > 665 and Debt_To_Income_Ratio <= 0.33 are consistently approved.

Conclusion:
Building an API-driven ingestion wrapper allows models to consume live credit updates directly. Decision trees provide human-auditable rule structures for regulatory credit compliance, directly identifying root feature split conditions without requiring complex feature transformation.