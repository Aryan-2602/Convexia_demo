import mlflow
from mlflow.tracking import MlflowClient
from utils.logger import logger

mlflow.set_tracking_uri("file://" + str((__file__).rsplit("/", 2)[0]) + "/mlruns")

client = MlflowClient()


def register_model_if_needed(model_name: str, run_id: str, artifact_path: str):
    """
    Registers a model in MLflow only if it's not already registered.

    Args:
        model_name (str): The name under which to register the model.
        run_id (str): The run ID that contains the model.
        artifact_path (str): The relative path to the model artifact in the run (e.g., 'ld50_model').
    """
    try:
        client.get_registered_model(model_name)
        logger.info(f"Model '{model_name}' is already registered in MLflow.")
    except mlflow.exceptions.RestException:
        logger.info(f"Model '{model_name}' not found. Registering now...")
        client.create_registered_model(model_name)

    # Check if model version for this run ID already exists
    versions = client.get_latest_versions(model_name)
    for v in versions:
        if v.run_id == run_id:
            logger.info(f"Run ID '{run_id}' is already associated with a version of '{model_name}'.")
            return

    # Register model version
    model_uri = f"runs:/{run_id}/{artifact_path}"
    client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=run_id
    )
    logger.success(f"Model '{model_name}' registered successfully.")
