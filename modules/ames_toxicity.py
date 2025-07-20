from transformers import pipeline
from utils.logger import logger
import torch
torch.set_num_threads(1)

logger.info("Loading Ames test toxicity model...")

ames_pipe = pipeline(
    "text-classification",
    model="ML4chemistry/Toxicity_Prediction_of_Ames_test",
    framework="pt",
    device=-1  # force CPU
)

def predict_ames_toxicity(smiles: str) -> dict:
    logger.info(f"Predicting Ames toxicity for: {smiles}")
    result = ames_pipe(smiles)[0]
    score = round(result["score"], 4)
    return score 

