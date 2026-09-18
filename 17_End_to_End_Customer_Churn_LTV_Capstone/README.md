"""
Week-4: Capstone Project: End-to-End Enterprise Customer Churn & Lifetime Value Engine
--------------------------

Platform: Google Colab / Python

Libraries: Pandas, NumPy, Scikit-Learn

"""
import pandas as pd
import numpy as np

# File 1: Customer Profile Logs (50 Records)
profile_data = {
    'Customer_ID': [f'CUST_{100+i}' for i in range(50)],
    'Account_Tier': ['Silver', 'Gold', 'Bronze', 'Platinum', 'Gold', 'Silver', 'Bronze', 'Gold', 'Platinum', 'Silver',
                     'Bronze', 'Gold', 'Silver', 'Platinum', 'Bronze', 'Gold', 'Silver', 'Platinum', 'Bronze', 'Gold',
                     'Silver', 'Bronze', 'Gold', 'Platinum', 'Silver', 'Bronze', 'Gold', 'Platinum', 'Silver', 'Bronze',
                     'Gold', 'Silver', 'Platinum', 'Bronze', 'Gold', 'Silver', 'Platinum', 'Bronze', 'Gold', 'Silver',
                     'Bronze', 'Gold', 'Platinum', 'Silver', 'Bronze', 'Gold', 'Platinum', 'Silver', 'Bronze', 'Gold'],
    'Age': [34, 45, np.nan, 29, 52, 38, 24, np.nan, 41, 31,
            27, 49, 36, 43, np.nan, 50, 33, 28, 22, 47,
            39, 26, 51, 42, 30, np.nan, 46, 35, 25, 48,
            37, 32, 44, 23, 53, 40, np.nan, 29, 47, 31,
            26, 50, 43, 34, 21, 49, 38, 27, 45, 33],
    'Region': ['North', 'South', 'West', 'East', 'North', 'South', 'West', 'East', 'North', 'South',
               'West', 'East', 'North', 'South', 'West', 'East', 'North', 'South', 'West', 'East',
               'North', 'South', 'West', 'East', 'North', 'South', 'West', 'East', 'North', 'South',
               'West', 'East', 'North', 'South', 'West', 'East', 'North', 'South', 'West', 'East',
               'North', 'South', 'West', 'East', 'North', 'South', 'West', 'East', 'North', 'South']
}

# File 2: Usage & Transaction Logs (50 Records)
usage_data = {
    'Customer_ID': [f'CUST_{100+i}' for i in range(50)],
    'Monthly_Charges_USD': [45.2, 89.0, 25.0, 110.5, 78.0, 35.0, 20.0, 95.0, 115.0, 40.0,
                            22.0, 85.0, 42.0, 105.0, 28.0, 92.0, 38.0, 112.0, 21.0, 88.0,
                            41.0, 24.0, 96.0, 108.0, 39.0, np.nan, 90.0, 118.0, 26.0, 84.0,
                            43.0, 37.0, 109.0, 19.0, 99.0, 44.0, 114.0, 23.0, 91.0, 36.0,
                            27.0, 93.0, 107.0, 40.0, 18.0, 94.0, 111.0, 33.0, 87.0, 38.0],
    'App_Sessions': [12, 35, 2, 85, 28, 8, 3, 38, 92, 10,
                    2, 31, 11, 78, 3, 36, 7, 88, 2, 33,
                    9, 3, 40, 75, 8, 1, 34, 90, 4, 32,
                    10, 8, 81, 1, 42, 9, 89, 3, 35, 11,
                    2, 37, 80, 12, 1, 38, 86, 7, 31, 9],
    'Support_Tickets': [1, 4, 0, 8, 3, 1, 0, 5, 9, 1,
                        0, 3, 1, 7, 0, 4, 1, 8, 0, 3,
                        1, 0, 4, 7, 1, 0, 3, 9, 0, 3,
                        1, 1, 8, 0, 5, 1, 8, 0, 4, 1,
                        0, 4, 7, 1, 0, 4, 8, 1, 3, 1]
}

# File 3: Retention Outcome Logs (50 Records)
outcome_data = {
    'Customer_ID': [f'CUST_{100+i}' for i in range(50)],
    'Churn': [0, 1, 0, 1, 1, 0, 0, 1, 1, 0,
              0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
              0, 0, 1, 1, 0, 0, 1, 1, 0, 1,
              0, 0, 1, 0, 1, 0, 1, 0, 1, 0,
              0, 1, 1, 0, 0, 1, 1, 0, 1, 0]
}
"""
Capstone Tasks & Individual Expected Outputs

TASK 1 — Multi-Source Relational Join: Perform sequential inner joins on df_profile, df_usage, 
and df_outcome using primary key Customer_ID across all 50 records.

Expected Output: Joined master dataset containing 50 rows and 8 initial columns.

TASK 2 — Grouped Imputation: Compute missing value counts and impute missing Age and Monthly_Charges_USD using median grouped by Account_Tier.

Expected Output: Missing values count = 0 post-imputation (Bronze Age Median = 25.5, Gold Spend Median = $89.00).

TASK 3 — Domain Feature Engineering: Create interaction feature Spend_Per_Session = Monthly_Charges_USD / (App_Sessions + 1e-5) to model engagement efficiency.
Expected Output: New feature column added to dataset.

TASK 4 — Categorical Encoding: Ordinal map Account_Tier (Bronze=1, Silver=2, Gold=3, Platinum=4) and
One-Hot encode Region dropping baseline 'East'. 
Expected Output: Discrete ordinal encoding + binary dummy columns (Region_North, Region_South, Region_West).

TASK 5 — IQR Outlier Capping: Identify outliers in Monthly_Charges_USD using 1.5*IQR bounds and cap upper extreme values.
Expected Output: Q1 = 33.5, Q3 = 94.75, Upper Bound= 186.625 (0 values capped).


TASK 6 — Train/Val/Test Splitting: Perform 60/20/20 stratified split on 50 records into Train (30), Validation (10), and Test (10) sets preventing data leakage.  
Expected Output: Train Shape = (30, 9), Val Shape = (10, 9), Test Shape = (10, 9).

TASK 7 — Class Imbalance Profiling: Evaluate class distribution across splits.  
Expected Output: Target distribution roughly 58% Non-Churn (0) vs 42% Churn (1).

TASK 8 — Scikit-Learn Pipeline Setup: Wrap StandardScaler and RandomForestClassifier inside an integrated Pipeline. 
Expected Output: Valid Scikit-Learn Pipeline object with preprocessing and estimator steps. 

TASK 9 — Decision Tree Base Model: Train a single baseline DecisionTreeClassifier on the training set.
Expected Output: Decision Tree baseline trained.

TASK 10 — Random Forest Training & Grid Tuning: Perform GridSearchCV over n_estimators [10, 30] and max_depth [3, 5] using 3-fold cross-validation.  
Expected Output: Optimal hyperparameters logged.

TASK 11 — Comprehensive Model Evaluation: Calculate test set Accuracy, Precision, Recall, F1-Score, and ROC-AUC.  
Expected Output: Test F1-Score >= 0.85, ROC-AUC>= 0.90.

TASK 12 — Confusion Matrix Breakdown: Extract True Positives, False Positives, True Negatives, and False Negatives from test predictions.  
Expected Output: Confusion matrix array breakdown.


Overall Output Summary
-------------------------
========== WEEK 4 CAPSTONE: CHURN PREDICTION PIPELINE ==========

Master Dataset Records     : 50
Raw Feature Columns        : 8
Processed Matrix Features  : 9

Preprocessing & Engineering:
- Imputed Values           : Age (5), Monthly_Charges_USD (1) via Account_Tier medians
- Interaction Features     : Created Spend_Per_Session metric
- Encoding                 : Account_Tier (Ordinal 1-4), Region (One-Hot dummy)

Data Split Ratios (60/20/20):
- Training Set             : 30 records
- Validation Set           : 10 records
- Test Set                 : 10 records

Model Tuning & Performance:
- Tuned Algorithm          : RandomForestClassifier
- Best Parameters          : max_depth=3, n_estimators=30
- Test Accuracy            : 100.00%
- Test F1-Score            : 1.00
- Test ROC-AUC             : 1.00

Pipeline Outcome: Machine Learning pipeline successfully trained and evaluated without data leakage.
"""