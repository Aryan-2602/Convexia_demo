import sys
import os

# Add project root (convexia_demo/) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from celery import Celery
from utils.predictor import run_all_models
from utils.logger import logger

celery = Celery(__name__, broker="redis://localhost:6379/0")

@celery.task(name="run_pipeline_task")
def run_pipeline_task(smiles: str):
    logger.info(f"Running prediction task for SMILES: {smiles}")
    predictions = run_all_models(smiles)
    logger.success(f"Predictions complete for {smiles}")
    return predictions
