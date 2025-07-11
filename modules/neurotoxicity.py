import numpy as np
from utils.logger import logger

def predict_neurotoxicity(smiles: str):
    logger.info(f"Predicting neurotoxicity for: {smiles}")

    # TODO: Add CONVERGE model
    prediction = np.random.uniform(0, 1)
    logger.debug(f"Stubbed neurotoxicity score: {prediction:.3f}")

    return prediction
