# Predicting SA Telco Churn with Eskom Load-Shedding Data

South African telco customers churn for reasons that go beyond the usual billing and contract factors. This project tests whether Eskom load-shedding exposure is a meaningful churn signal by joining daily shed-hour schedules per area to a telco churn dataset, then running the full training workflow through a SageMaker Pipeline managed with CDK.

The infrastructure lifecycle (pipeline registration, model registry setup and cleanup on destroy) is fully automated. No manual AWS console steps are needed after `cdk deploy`.

---

## Project DAG structure

![DAG structure](images/dag_structure.png)

## What this does

The pipeline runs five steps in sequence:

1. **Preprocessing** — joins the churn dataset with a location-to-area-code map and an Eskom load-shedding schedule, encodes categoricals, scales numeric features and writes train/validation/test splits
2. **Training** — trains an XGBoost binary classifier using the SageMaker built-in container
3. **Evaluation** — scores the model on the test split and writes an `evaluation.json` report with AUC and accuracy
4. **Condition check** — registers the model in SageMaker Model Registry if AUC meets the threshold, otherwise fails the execution with the actual AUC in the error message

Infrastructure is managed with CDK across two top-level stacks: `StorageStack` (S3 buckets and data uploads) and `PipelineStack` (IAM roles, model registry, pipeline registration via CloudFormation custom resource).

---

## Results

The core question this project tests is whether Eskom load-shedding exposure adds predictive signal for SA telco churn, on top of the standard billing and contract features.

The XGBoost model was evaluated on the held-out test split and produced the following metrics:

| Metric | Value |
|---|---|
| AUC | 0.833 |
| Accuracy | 0.786 |

An AUC of 0.833 means the model can correctly rank a churner above a non-churner 83% of the time, which is a reasonable result for a churn model. The model cleared the 0.7 threshold and was registered in the SageMaker Model Registry with `PendingManualApproval` status.

The data is synthetic, so these numbers cannot be taken as evidence that load-shedding is a real churn driver. What the pipeline demonstrates is that once such a feature is available, built from actual Eskom schedules joined to a real subscriber base, the infrastructure is ready to test it at scale.

---

## Data architecture

This project follows a medallion lakehouse pattern where data moves through layers before reaching the ML pipeline.

In a production setup, raw source data (bronze) would be processed by an ETL job and land in the silver layer as cleansed, joined and lightly transformed datasets. The ML pipeline picks up from silver and does the ML-specific feature engineering from there.

The silver bucket holds three synthetically generated files that represent what an ETL pipeline would have produced:

- `telco_churn_sa_loc.csv` — the core churn dataset, adapted from the [IBM Telco customer churn dataset on Kaggle](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset/data). It uses a `location_id` column instead of a raw area code, meaning the ETL already resolved geographic identifiers to an internal key
- `location_area_map.csv` — a reference table mapping `location_id` to `area_code`, used by the preprocessing step to resolve the join
- `eskom_schedule_daily.csv` — load-shedding schedules aggregated to daily shed hours per area code, representing an external data source already cleaned and structured by an ETL job

CDK uploads the silver files under a Hive-style prefix (`year=/month=/day=/`) computed from the date of deployment. This mirrors how a real ETL job would partition output by run date. The ML bucket receives the preprocessing scripts without a prefix.

The preprocessing step (step 1 of the pipeline) does the ML-specific work: joins the three datasets, engineers the `loadshedding_exposure_hrs` feature, drops leakage columns, encodes categoricals, scales numerics and produces the train/validation/test splits. That output lands in the ML bucket.

Two separate IAM roles are provisioned: one for preprocessing jobs and one for training jobs. Both currently use `AmazonSageMakerFullAccess`, with read/write on the relevant buckets granted on top. The code notes where least-privilege scoping should be applied in production.

The SageMaker Pipeline itself is managed as a CloudFormation resource using `AwsCustomResource`. The pipeline definition JSON is generated at synth time by calling the SageMaker Pipelines SDK, then passed to CloudFormation hooks: `createPipeline` on first deploy, `updatePipeline` on redeploy if the definition changes and `deletePipeline` on destroy. This means the pipeline stays in sync with the CDK code without any manual registration steps.

---

## Prerequisites

- AWS CLI configured with credentials
- Node.js (for CDK CLI)
- Python 3.9+
- CDK bootstrapped in your target account/region (`cdk bootstrap`)

```bash
pip install -r requirements_cdk.txt
npm install -g aws-cdk
```

---

## Deploy

```bash
cdk deploy --all
```

This deploys both stacks. `PipelineStack` registers the SageMaker Pipeline in your account via a CloudFormation custom resource, so the pipeline exists in SageMaker as soon as the deploy completes. To start an execution, trigger it from the SageMaker Studio console, the AWS CLI, or run `python start_pipeline.py`.

---

## Destroy

Before running `cdk destroy`, you must manually delete any registered model versions from the Model Registry. CloudFormation cannot delete a model package group that still contains versions, so the destroy will fail if you skip this step.

To delete all model versions for this project, run:

```bash
aws sagemaker list-model-packages --model-package-group-name churn-model-group \
  --query "ModelPackageSummaryList[*].ModelPackageArn" --output text | \
  tr '\t' '\n' | \
  xargs -I {} aws sagemaker delete-model-package --model-package-name {}
```

Then destroy the stacks:

```bash
cdk destroy --all
```

The SageMaker Pipeline is deleted automatically via the CloudFormation custom resource on destroy.

---

## Pipeline parameters

These can be overridden at execution time from the console or CLI:

| Parameter | Default |
|---|---|
| `ProcessingInstanceType` | `ml.t3.medium` |
| `TrainingInstanceType` | `ml.m5.large` |
| `ModelApprovalStatus` | `PendingManualApproval` |
| `AucThreshold` | `0.7` |

---

## Limitations

**No automatic execution on deploy.** The pipeline is registered during `cdk deploy` but executions are triggered manually. Level 2 MLOps would add an event-based trigger (e.g. S3 upload or EventBridge schedule).

**Manual model approval.** New model versions land in the registry with `PendingManualApproval`. There is no automated approval gate or downstream deployment step.

**Broad IAM permissions.** Both the preprocessing and training roles use `AmazonSageMakerFullAccess` as a base policy with bucket-level grants on top. Production would separate these further and scope each role down to only the actions each job type actually needs.

**All data is lost on destroy.** Both S3 buckets are created with `RemovalPolicy.DESTROY` and `auto_delete_objects=True`. Running `cdk destroy` deletes all training data, model artifacts and preprocessing outputs.

**No inference endpoint.** The pipeline ends at model registration. Deploying an endpoint is a separate step not covered here.

**Customer ID is not encrypted.** The `CustomerID` column is dropped during preprocessing rather than properly encrypted. AWS Secrets Manager integration for this is not yet implemented.

**Silver prefix is fixed at deploy time.** The Hive partition prefix on the silver bucket is computed from the date `cdk deploy` runs. If the stack is redeployed on a different day, a new prefix is created and the old data stays under the previous path. There is no mechanism to point the pipeline at a specific partition.

**Synthetic SA data.** The Eskom load-shedding and location datasets are constructed for this project and do not reflect real network or operational data.

**Single region.** No cross-region replication or disaster recovery is set up.
# aws-telecom-churn-eskom
