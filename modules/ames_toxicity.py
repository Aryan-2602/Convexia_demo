from transformers import pipeline
from utils.logger import logger
import torch
import mlflow
from pathlib import Path

torch.set_num_threads(1)

logger.info("Loading Ames test toxicity model...")

ames_pipe = pipeline(
    "text-classification",
    model="ML4chemistry/Toxicity_Prediction_of_Ames_test",
    framework="pt",
    device=-1  # force CPU
)

mlflow.set_tracking_uri("file://" + str(Path(__file__).resolve().parent.parent / "mlruns"))
mlflow.set_experiment("ames_toxicity")

def predict_ames_toxicity(smiles: str) -> float:
    logger.info(f"Predicting Ames toxicity for: {smiles}")
    with mlflow.start_run(run_name="ames_inference",nested=True):
        mlflow.set_tag("model", "ames_model")
        result = ames_pipe(smiles)[0]
        score = round(result["score"], 4)

        mlflow.log_param("model_name", "ML4chemistry/Toxicity_Prediction_of_Ames_test")
        mlflow.log_metric("ames_score", score)

        return score

