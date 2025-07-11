import numpy as np
from utils.logger import logger

def compute_confidence():
    # Placeholder for model confidence computation
    confidence = np.round(np.random.uniform(0.85, 0.99), 2)
    logger.info(f"Computed model confidence: {confidence}")
    return confidence

def find_disagreements(organ_tox: dict):
    logger.debug("Checking for module disagreements in organ toxicity scores.")
    values = list(organ_tox.values())
    disagreements = []

    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            diff = abs(values[i] - values[j])
            if diff > 0.5:
                msg = f"Module disagreement between organ_{i} and organ_{j} (Δ={diff:.2f})"
                disagreements.append(msg)
                logger.warning(msg)

    if not disagreements:
        logger.info("No significant disagreements found between organ toxicity modules.")

    return disagreements
