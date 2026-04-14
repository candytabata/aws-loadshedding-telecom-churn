import sagemaker, boto3, os
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString, ParameterFloat
from sagemaker.workflow.steps import TrainingStep, ProcessingStep
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.inputs import TrainingInput
from sagemaker.estimator import Estimator
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.model_metrics import MetricsSource, ModelMetrics
# from sagemaker.workflow.check_job_config import CheckJobConfig
# from sagemaker.workflow.quality_check_step import QualityCheckStep, DataQualityCheckConfig
# from sagemaker.model_monitor.dataset_format import DatasetFormat

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
SKLEARN_VERSION = "0.20.0"
XGBOOST_VERSION = "1.3-1"


def get_pipeline(
    region: str,
    role: str,
    silver_bucket_name: str,
    ml_bucket_name: str,
    pipeline_name: str = "churn-pipeline",
    model_package_group_name: str = "churn-model-group",
    base_job_prefix: str = "sa-churn",
) -> Pipeline:

    # ── Session ───────────────────────────────────────────────────────────────
    boto_session      = boto3.Session(region_name=region)
    sagemaker_client  = boto_session.client("sagemaker")
    pipeline_session  = PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        # default_bucket=ml_bucket_name,
    )
    # Prevent the SDK from creating the bucket at synth time.
    # Session.default_bucket() only skips creation if _default_bucket is already set.
    pipeline_session._default_bucket = ml_bucket_name

    # ── Parameters ────────────────────────────────────────────────────────────
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType", default_value="ml.t3.medium"
    )
    training_instance_type = ParameterString(
        name="TrainingInstanceType", default_value="ml.m5.large"
    )
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )
    auc_threshold = ParameterFloat(name="AucThreshold", default_value=0.7)

    # ── Steps ───────────────────────────────────────────────────────────────
    # Processing step: data preprocessing (Python script)
    sklearn_processor = SKLearnProcessor(
        framework_version=SKLEARN_VERSION,
        role=role,
        instance_type=processing_instance_type,
        instance_count=1,
        volume_size_in_gb=5,
        base_job_name=f"{base_job_prefix}-preprocessing",
        sagemaker_session=pipeline_session,
    )

    processing_step = ProcessingStep(
        name="ChurnDataPreprocessing",
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(
                source=f"s3://{silver_bucket_name}/",
                destination="/opt/ml/processing/input/silver",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="splits",
                source="/opt/ml/processing/output",
                destination=Join(on="/", values=[
                    f"s3://{ml_bucket_name}",
                    base_job_prefix,
                    ExecutionVariables.PIPELINE_EXECUTION_ID,
                    "preprocessing",
                ]),
            ),
        ],
        code=f"s3://{ml_bucket_name}/scripts/preprocessing/preprocess.py",
    )

    # Data quality check step — commented out due to Spark overhead costs.
    # Uses SageMaker Model Monitor container which runs a full Spark job internally,
    # making it slow and expensive on small instances. Re-enable when cost is not a concern.
    #
    # check_job_config = CheckJobConfig(
    #     role=role,
    #     instance_type=training_instance_type,
    #     instance_count=1,
    #     volume_size_in_gb=5,
    #     max_runtime_in_seconds=3600,
    #     sagemaker_session=pipeline_session,
    # )
    #
    # data_quality_check_config = DataQualityCheckConfig(
    #     baseline_dataset=Join(on="/", values=[
    #         processing_step.properties.ProcessingOutputConfig.Outputs["splits"].S3Output.S3Uri,
    #         "train.csv",
    #     ]),
    #     dataset_format=DatasetFormat.csv(header=False),
    #     output_s3_uri=Join(on="/", values=[
    #         f"s3://{ml_bucket_name}",
    #         base_job_prefix,
    #         ExecutionVariables.PIPELINE_EXECUTION_ID,
    #         "data-quality-baseline",
    #     ]),
    # )
    #
    # data_quality_check_step = QualityCheckStep(
    #     name="ChurnDataQualityBaseline",
    #     quality_check_config=data_quality_check_config,
    #     check_job_config=check_job_config,
    #     skip_check=True,
    #     register_new_baseline=True,
    #     depends_on=["ChurnDataPreprocessing"],
    # )

    # Training step: train XGBoost model using SageMaker built-in XGBoost container
    xgboost_estimator = Estimator(
        image_uri=sagemaker.image_uris.retrieve(
            framework="xgboost",
            region=region,
            version=XGBOOST_VERSION,
            image_scope="training",
        ),
        role=role,
        instance_count=1,
        volume_size=5,
        instance_type=training_instance_type,
        base_job_name=f"{base_job_prefix}-training",
        sagemaker_session=pipeline_session,
        output_path=f"s3://{ml_bucket_name}/output/",
        hyperparameters={
            "objective":        "binary:logistic",
            "num_round":        "100",
            "max_depth":        "6",
            "eta":              "0.3",
            "subsample":        "0.8",
            "colsample_bytree": "0.8",
            "eval_metric":      "auc",
        },
    )
    training_step = TrainingStep(
        name="ChurnModelTraining",
        estimator=xgboost_estimator,
        inputs={
            "train": TrainingInput(
                s3_data=Join(on="/", values=[
                    processing_step.properties.ProcessingOutputConfig.Outputs["splits"].S3Output.S3Uri,
                    "train.csv",
                ]),
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=Join(on="/", values=[
                    processing_step.properties.ProcessingOutputConfig.Outputs["splits"].S3Output.S3Uri,
                    "validation.csv",
                ]),
                content_type="text/csv",
            ),
        },
        # depends_on=["ChurnDataQualityBaseline"],
    )

    # Evaluation processor — reuses the XGBoost image so xgboost is available without extra installs
    eval_processor = ScriptProcessor(
        image_uri=sagemaker.image_uris.retrieve(
            framework="xgboost",
            region=region,
            version=XGBOOST_VERSION,
            image_scope="training",
        ),
        command=["python3"],
        role=role,
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{base_job_prefix}-evaluation",
        sagemaker_session=pipeline_session,
    )

    #Evaluate model performance and conditionally register the model if AUC ≥ threshold
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )

    evaluation_step = ProcessingStep(
        name="ChurnModelEvaluation",
        processor=eval_processor,
        inputs=[
            ProcessingInput(
                source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/input/model",
            ),
            ProcessingInput(
                source=Join(on="/", values=[
                    processing_step.properties.ProcessingOutputConfig.Outputs["splits"].S3Output.S3Uri,
                    "test.csv",
                ]),
                destination="/opt/ml/processing/input/test",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/output/evaluation",
            )
        ],
        code=f"s3://{ml_bucket_name}/scripts/evaluation/evaluate.py",
        property_files=[evaluation_report],
    )

    # Condition step to check if AUC ≥ threshold
    condition_step = ConditionStep(
        name="CheckAUCThreshold",
        conditions=[
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(
                    step_name=evaluation_step.name,
                    property_file=evaluation_report,
                    json_path="binary_classification_metrics.auc.value",
                ),
                right=auc_threshold,
            )
        ],
        if_steps=[
            RegisterModel(
                name="RegisterChurnModel",
                estimator=xgboost_estimator,
                model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                content_types=["text/csv"],
                response_types=["text/csv"],
                inference_instances=["ml.t2.medium", "ml.m5.large"],
                transform_instances=["ml.m5.large"],
                model_package_group_name=model_package_group_name,
                approval_status=model_approval_status,
                model_metrics=ModelMetrics(
                    model_statistics=MetricsSource(
                        s3_uri=Join(on="/", values=[
                            evaluation_step.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri,
                            "evaluation.json",
                        ]),
                        content_type="application/json",
                    )
                ),
            )
        ],
        else_steps=[
            FailStep(
                name="FailChurnModelEvaluation",
                error_message=Join(on=" ", values=[
                    "Model performance did not meet the threshold.",
                    "AUC was",
                    JsonGet(
                        step_name=evaluation_step.name,
                        property_file=evaluation_report,
                        json_path="binary_classification_metrics.auc.value",
                    ),
                    "but needed to be at least",
                    auc_threshold,
                ]),
            )
        ],
    )

    # ── Pipeline ───────────────────────────────────────────────────────────────
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            training_instance_type,
            model_approval_status,
            auc_threshold,
        ],
        steps=[
            processing_step,
            # data_quality_check_step,
            training_step,
            evaluation_step,
            condition_step,
        ],
        sagemaker_session=pipeline_session,
    )
    return pipeline
