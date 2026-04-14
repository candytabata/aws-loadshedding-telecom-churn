"""
start_pipeline.py — registers the SA Churn SageMaker Pipeline and starts an execution.

Reads infrastructure values from AWS directly (IAM role by name, silver bucket
from CloudFormation outputs) so no values need to be hardcoded.

Usage:
    python start_pipeline.py [--region us-east-1] [--no-start]
"""

import argparse
import boto3

from pipeline_definition import get_pipeline

PIPELINE_ROLE_NAME    = "churn-sagemaker-processing-role"
STORAGE_STACK_NAME    = "StorageStack"
SILVER_BUCKET_OUT_KEY = "SilverBucketName"


def get_role_arn(iam_client) -> str:
    return iam_client.get_role(RoleName=PIPELINE_ROLE_NAME)["Role"]["Arn"]


def get_silver_bucket_name(cf_client) -> str:
    outputs = cf_client.describe_stacks(StackName=STORAGE_STACK_NAME)["Stacks"][0]["Outputs"]
    for output in outputs:
        if output["OutputKey"] == SILVER_BUCKET_OUT_KEY:
            return output["OutputValue"]
    raise RuntimeError(
        f"Output '{SILVER_BUCKET_OUT_KEY}' not found in stack '{STORAGE_STACK_NAME}'. "
        "Run 'cdk deploy --all' first."
    )


def main():
    parser = argparse.ArgumentParser(description="Register and start the SA Churn SageMaker Pipeline")
    parser.add_argument("--region",   default=None, help="AWS region (defaults to boto3 session region)")
    parser.add_argument("--no-start", action="store_true", help="Register the pipeline only, do not start an execution")
    args = parser.parse_args()

    session    = boto3.Session()
    region     = args.region or session.region_name
    account_id = session.client("sts").get_caller_identity()["Account"]

    iam_client = session.client("iam",            region_name=region)
    cf_client  = session.client("cloudformation", region_name=region)

    role_arn           = get_role_arn(iam_client)
    silver_bucket_name = get_silver_bucket_name(cf_client)
    ml_bucket_name     = f"ml-data-{account_id}-{region}"

    print(f"[config] region         = {region}")
    print(f"[config] role           = {role_arn}")
    print(f"[config] silver_bucket  = {silver_bucket_name}")
    print(f"[config] ml_bucket      = {ml_bucket_name}")

    pipeline = get_pipeline(
        region=region,
        role=role_arn,
        silver_bucket_name=silver_bucket_name,
        ml_bucket_name=ml_bucket_name,
    )

    pipeline.upsert(role_arn=role_arn)
    print("[pipeline] registered/updated successfully")

    if not args.no_start:
        execution = pipeline.start()
        print(f"[pipeline] execution started: {execution.arn}")


if __name__ == "__main__":
    main()
