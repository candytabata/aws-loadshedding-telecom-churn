#!/usr/bin/env python3

import aws_cdk as cdk

from stacks.ml_stack import StorageStack, PipelineStack


app = cdk.App()

storage_stack = StorageStack(app, "StorageStack")

pipeline_stack = PipelineStack(
    app, "PipelineStack",
    silver_bucket=storage_stack.silver_bucket.bucket_name,
    ml_bucket=storage_stack.ml_bucket.bucket_name,
    model_package_group_name="churn-model-group",
)

app.synth()
