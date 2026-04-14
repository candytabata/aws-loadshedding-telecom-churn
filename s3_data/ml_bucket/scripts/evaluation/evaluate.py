"""
SageMaker Processing script — SA Churn Model Evaluation

Loads the XGBoost model artifact and the test split produced by preprocess.py,
computes AUC and accuracy, then writes an evaluation report in the format
expected by SageMaker Model Registry.

Expected mounts
---------------
  /opt/ml/processing/input/test/test.csv      — test split (target first, no header)
  /opt/ml/processing/input/model/model.tar.gz — XGBoost model artifact from training step

Output
------
  /opt/ml/processing/output/evaluation/evaluation.json
"""

import json
import os
import tarfile

import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score

INPUT_TEST  = "/opt/ml/processing/input/test/test.csv"
INPUT_MODEL = "/opt/ml/processing/input/model/model.tar.gz"
OUTPUT_DIR  = "/opt/ml/processing/output/evaluation"
MODEL_DIR   = "/opt/ml/processing/model"


#  Load test data 
test_df = pd.read_csv(INPUT_TEST, header=None)
y_test  = test_df.iloc[:, 0].values.astype(int)
X_test  = test_df.iloc[:, 1:].values.astype(float)
print(f"[test] {len(test_df):,} rows, {X_test.shape[1]} features")


# Extract and load model 
os.makedirs(MODEL_DIR, exist_ok=True)
with tarfile.open(INPUT_MODEL, "r:gz") as tar:
    tar.extractall(MODEL_DIR)

model = xgb.Booster()
model.load_model(os.path.join(MODEL_DIR, "xgboost-model"))
print("[model] loaded xgboost-model")


# Predict and score 
dmatrix  = xgb.DMatrix(X_test)
y_prob   = model.predict(dmatrix)
y_pred   = (y_prob >= 0.5).astype(int)

auc      = float(roc_auc_score(y_test, y_prob))
accuracy = float(accuracy_score(y_test, y_pred))
print(f"[eval] AUC={auc:.4f}  Accuracy={accuracy:.4f}")


# Write evaluation report 
os.makedirs(OUTPUT_DIR, exist_ok=True)

report = {
    "binary_classification_metrics": {
        "auc": {
            "value": auc,
            "standard_deviation": 0.0,
        },
        "accuracy": {
            "value": accuracy,
            "standard_deviation": 0.0,
        },
    }
}

output_path = os.path.join(OUTPUT_DIR, "evaluation.json")
with open(output_path, "w") as f:
    json.dump(report, f, indent=2)

print(f"[output] wrote {output_path}")
