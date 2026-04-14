"""
SageMaker Processing script — SA Churn ML Preprocessing

Expected channel mounts:
  /opt/ml/processing/input/silver/   — Silver bucket root (Parquet churn data under
                                        telecom_churn/ and eskom_schedule_daily.csv
                                        under the hive-dated prefix)

Output:
  /opt/ml/processing/output/train.csv
  /opt/ml/processing/output/validation.csv
  /opt/ml/processing/output/test.csv

  Each file: target (Churn Value) as first column, no header.
  Compatible with SageMaker built-in XGBoost.
"""

import glob
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

INPUT_SILVER = "/opt/ml/processing/input/silver"
OUTPUT_DIR   = "/opt/ml/processing/output"

RANDOM_STATE = 42

# Load Silver churn data (with location_id but no area_code yet)
churn_matches = glob.glob(os.path.join(INPUT_SILVER, "**/telco_churn_sa_loc.csv"), recursive=True)
churn_df = pd.read_csv(churn_matches[0])
print(f"[silver] loaded {len(churn_df):,} rows, {churn_df.shape[1]} columns")

# Resolve location_id → area_code via location_area_map
loc_map_matches = glob.glob(os.path.join(INPUT_SILVER, "**/location_area_map.csv"), recursive=True)
location_map_df = pd.read_csv(loc_map_matches[0])[["location_id", "area_code"]]
churn_df = churn_df.merge(location_map_df, on="location_id", how="left")
churn_df.drop(columns=["location_id"], inplace=True)

missing_area = churn_df["area_code"].isna().sum()
if missing_area:
    print(f"[warn] {missing_area} rows have no area_code match in location_area_map")
print(f"[location] resolved area_code for {len(churn_df):,} rows")


# Aggregate Eskom schedule → one scalar feature per area_code
# eskom_schedule_daily has one row per (area_code, shed window) across 182 days.
# Collapse it to avg daily load-shedding hours per area.
eskom_matches = glob.glob(os.path.join(INPUT_SILVER, "**/eskom_schedule_daily.csv"), recursive=True)
eskom_df = pd.read_csv(eskom_matches[0])

eskom_agg = (
    eskom_df
    .groupby("area_code", as_index=False)["shed_hrs"]
    .sum()
    .assign(loadshedding_exposure_hrs=lambda x: x["shed_hrs"] / 182)
    [["area_code", "loadshedding_exposure_hrs"]]
)
print(f"[eskom] aggregated to {len(eskom_agg)} area records")


# Join Eskom feature onto churn (area_code now resolved from location_area_map)
df = churn_df.merge(eskom_agg, on="area_code", how="left")
df.drop(columns=["area_code"], inplace=True)

missing_exposure = df["loadshedding_exposure_hrs"].isna().sum()
if missing_exposure:
    print(f"[warn] {missing_exposure} rows have no Eskom match — filling with 0")
    df["loadshedding_exposure_hrs"].fillna(0, inplace=True)


# Drop metadata / leakage columns
DROP_COLS = [
    "CustomerID",    # hashed PII — not a predictive feature
    "Lat Long",      # raw coordinate string — redundant after area_code join
    "Latitude",
    "Longitude",
    "location_id",   # resolved to area_code
    "area_code",     # geographic ID — exposure_hrs already captures its signal
    "Churn Label",   # string duplicate of the target (leakage)
    "Churn Score",   # pre-computed score — direct leakage
    "CLTV",          # pre-computed value derived from churn probability — leakage
    "Churn Reason",  # only known after the customer has churned (leakage)
]
df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])


# Separate target
TARGET = "Churn Value"
y = df[TARGET].astype(int)
X = df.drop(columns=[TARGET])


# Encode binary Yes/No columns → 0/1
BINARY_YES_NO = [
    "Partner", "Dependents", "Phone Service", "Paperless Billing",
]
for col in BINARY_YES_NO:
    if col in X.columns:
        X[col] = X[col].map({"Yes": 1, "No": 0})

# Gender
X["Gender"] = X["Gender"].map({"Male": 1, "Female": 0})

# Senior Citizen arrives as Yes/No 
X["Senior Citizen"] = X["Senior Citizen"].map({"Yes": 1, "No": 0})

# Three-value service columns: "No <service>" means the customer doesn't have
# that base service — treated the same as "No" (i.e. not subscribed = 0).
THREE_VALUE_COLS = ["Multiple Lines", "Device Protection", "Tech Support", "ups_included"]
for col in THREE_VALUE_COLS:
    if col in X.columns:
        X[col] = X[col].apply(lambda v: 1 if v == "Yes" else 0)

# data_cap: Uncapped=1, Capped/No Internet Service=0
X["data_cap"] = X["data_cap"].apply(lambda v: 1 if v == "Uncapped" else 0)


# One-hot encode remaining categoricals
OHE_COLS = ["internet_access_type_sa", "payment_method_sa", "contract_type_sa"]
X = pd.get_dummies(X, columns=[c for c in OHE_COLS if c in X.columns], drop_first=True)


# Scale numeric columns
NUM_COLS = [
    "Tenure Months",
    "Monthly Charge ZAR",
    "Total Charges ZAR",
    "loadshedding_exposure_hrs",
]
num_present = [c for c in NUM_COLS if c in X.columns]
scaler = StandardScaler()
X[num_present] = scaler.fit_transform(X[num_present])


# Verify no nulls remain
null_counts = X.isnull().sum()
if null_counts.any():
    print(f"[warn] nulls remaining:\n{null_counts[null_counts > 0]}")


# Train / validation / test split (70 / 15 / 15, stratified)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
)

print(f"[split] train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}")
print(f"[churn rate] train={y_train.mean():.3f}  val={y_val.mean():.3f}  test={y_test.mean():.3f}")


# Write outputs — target first column, no header
os.makedirs(OUTPUT_DIR, exist_ok=True)

def write_split(X_split, y_split, name: str) -> None:
    out = pd.concat(
        [y_split.reset_index(drop=True), X_split.reset_index(drop=True)],
        axis=1,
    )
    path = os.path.join(OUTPUT_DIR, f"{name}.csv")
    out.to_csv(path, index=False, header=False)
    print(f"[output] wrote {path}  ({len(out):,} rows × {out.shape[1]} cols)")

write_split(X_train, y_train, "train")
write_split(X_val,   y_val,   "validation")
write_split(X_test,  y_test,  "test")
