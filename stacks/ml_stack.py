import os, sys, boto3
from datetime import datetime
from constructs import Construct
from aws_cdk import (
    Stack,
    NestedStack,
    RemovalPolicy,
    CfnOutput,
    aws_s3 as s3,
    aws_s3_deployment as s3_deploy,
    aws_iam as iam,
    aws_sagemaker as sagemaker,
    custom_resources as cr
)

# Allow ml_stack.py (inside stacks/) to import pipeline_definition.py (one level up)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

class StorageStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Create S3 buckets 

        #  Silver bucket with data engineered datasets in csv, organised by Hive partitions (year/month/day)
        self.silver_bucket = s3.Bucket(self, "Silver-amzn-2026-04",
            bucket_name=f"silver-{self.account}-{self.region}",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            versioned=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED
        )

        # S3 bucket for SageMaker Jobs.  
        self.ml_bucket = s3.Bucket(
            self, "MlDataBucket",
            bucket_name=f"ml-data-{self.account}-{self.region}",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=RemovalPolicy.DESTROY,  
            auto_delete_objects=True, 
        )

        today = datetime.today()
        hive_prefix = f"year={today.year}/month={today.month:02d}/day={today.day:02d}/"

        # Uploads

        s3_deploy.BucketDeployment(self, "UploadTelecomChurnDataSilver",
            sources=[s3_deploy.Source.asset("s3_data/silver/")],
            destination_bucket=self.silver_bucket,
            destination_key_prefix=hive_prefix
        )

        s3_deploy.BucketDeployment(self, "UploadTelecomChurnDataMlData",
            sources=[s3_deploy.Source.asset("s3_data/ml_bucket/")],
            destination_bucket=self.ml_bucket
        )

        CfnOutput(self, "SilverBucketName", value=self.silver_bucket.bucket_name)


class PreprocessingStack(NestedStack):

    def __init__(self, scope: Construct, construct_id: str, silver_bucket: str, ml_bucket: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        
        # Reference the existing bucket by name
        silver_bucket = s3.Bucket.from_bucket_name(
            self, "SilverBucket", silver_bucket
        )

        ml_bucket = s3.Bucket.from_bucket_name(
            self, "MlBucket", ml_bucket
        )

        # IAM role assumed by SageMaker Processing jobs
        self.sagemaker_processing_role = iam.Role(
            self, "SageMakerProcessingRole",
            role_name="churn-sagemaker-processing-role",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            description="Allows SageMaker Processing jobs to read Silver data and write preprocessed ML features",
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonSageMakerFullAccess" )
            ]
        )

        silver_bucket.grant_read(self.sagemaker_processing_role)
        ml_bucket.grant_read_write(self.sagemaker_processing_role) # Production code should separate read/write permissions and apply least privilege.



class TrainingStack(NestedStack):

    def __init__(self, scope: Construct, construct_id: str, ml_bucket: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # IAM role assumed by SageMaker Training jobs
        self.sagemaker_training_role = iam.Role(
            self, "SageMakerTrainingRole",
            role_name="churn-sagemaker-training-role",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            description="Allows SageMaker Training jobs to read preprocessed ML features and write model artifacts",
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonSageMakerFullAccess")
            ]
        )

        ml_bucket = s3.Bucket.from_bucket_name(
            self, "MlBucket", ml_bucket
        )
        
        ml_bucket.grant_read_write(self.sagemaker_training_role)


class ModelRegistryStack(NestedStack):

    def __init__(self, scope: Construct, construct_id: str, model_package_group_name: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # IAM role assumed by SageMaker Model Registry operations (e.g. registering model packages)
        self.sagemaker_model_registry_role = iam.Role(
            self, "SageMakerModelRegistryRole",
            role_name="churn-sagemaker-model-registry-role",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            description="Allows SageMaker to perform Model Registry operations like registering model packages",
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonSageMakerFullAccess")
            ]
        )

        # Model package group for versioning and approving churn models
        model_group = sagemaker.CfnModelPackageGroup(
            self, "ChurnModelPackageGroup",
            model_package_group_name=model_package_group_name,
            model_package_group_description="SA churn XGBoost model versions",
        )
        model_group.apply_removal_policy(RemovalPolicy.DESTROY)

        self.model_package_group_name = model_package_group_name

class PipelineStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, silver_bucket: str, ml_bucket: str, model_package_group_name: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.preprocessing_stack = PreprocessingStack(
            self, "PreprocessingStack",
            silver_bucket=silver_bucket,
            ml_bucket=ml_bucket,
        )

        self.training_stack = TrainingStack(
            self, "TrainingStack",
            ml_bucket=ml_bucket,
        )

        self.model_registry_stack = ModelRegistryStack(
            self, "ModelRegistryStack",
            model_package_group_name=model_package_group_name,
        )

        # Resolve concrete values at synth time — boto3 runs during cdk deploy
        # so AWS credentials are available. This mirrors how start_pipeline.py works.
        boto_session = boto3.Session()
        region      = boto_session.region_name
        account_id  = boto_session.client("sts").get_caller_identity()["Account"]
        role_arn    = f"arn:aws:iam::{account_id}:role/churn-sagemaker-processing-role"

        # CDK tokens are unresolved Python strings at synth time — construct concrete
        # bucket names from boto3 values instead of passing the token objects.
        ml_bucket_name_concrete    = f"ml-data-{account_id}-{region}"
        silver_bucket_name_concrete = f"silver-{account_id}-{region}"

        from pipeline_definition import get_pipeline
        pipeline = get_pipeline(
            region=region,
            role=role_arn,
            silver_bucket_name=silver_bucket_name_concrete,
            ml_bucket_name=ml_bucket_name_concrete,
            model_package_group_name=model_package_group_name,
        )
        pipeline_definition_json = pipeline.definition()

        # Register the pipeline on deploy, update on re-deploy, delete on destroy
        cr.AwsCustomResource(
            self, "RegisterPipeline",
            on_create=cr.AwsSdkCall(
                service="SageMaker",
                action="createPipeline",
                parameters={
                    "PipelineName":       "churn-pipeline",
                    "PipelineDefinition": pipeline_definition_json,
                    "RoleArn":            role_arn,
                },
                physical_resource_id=cr.PhysicalResourceId.of("churn-pipeline"),
            ),
            on_update=cr.AwsSdkCall(
                service="SageMaker",
                action="updatePipeline",
                parameters={
                    "PipelineName":       "churn-pipeline",
                    "PipelineDefinition": pipeline_definition_json,
                    "RoleArn":            role_arn,
                },
                physical_resource_id=cr.PhysicalResourceId.of("churn-pipeline"),
            ),
            on_delete=cr.AwsSdkCall(
                service="SageMaker",
                action="deletePipeline",
                parameters={"PipelineName": "churn-pipeline"},
                physical_resource_id=cr.PhysicalResourceId.of("churn-pipeline"),
            ),
            policy=cr.AwsCustomResourcePolicy.from_statements([
                iam.PolicyStatement(
                    actions=[
                        "sagemaker:CreatePipeline",
                        "sagemaker:UpdatePipeline",
                        "sagemaker:DeletePipeline",
                    ],
                    resources=["*"],
                ),
                iam.PolicyStatement(
                    actions=["iam:PassRole"],
                    resources=[role_arn],
                ),
            ]),
        )
