import sys
import os

# Add project root (convexia_demo/) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.celery_utils import celery_app
from run_pipeline import run_toxicity_pipeline
from utils.logger import logger

@celery_app.task(name="toxicity_pipeline.run")
def run_pipeline_task(smiles: str):
    logger.info(f"Received SMILES for prediction: {smiles}")
    result = run_toxicity_pipeline(smiles)
    logger.info(f"Pipeline completed for SMILES: {smiles}")
    return result
