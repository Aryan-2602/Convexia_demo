import yaml
from utils.logger import logger
from pathlib import Path

# Load weights from YAML file
def load_weights():
    weights_path = Path(__file__).resolve().parent.parent / "config" / "scoring_weights.yaml"
    with open(weights_path, "r") as f:
        config = yaml.safe_load(f)
    return config["weights"]

# Load weights once at module-level
WEIGHTS = load_weights()


def compute_composite_score(values: dict):
    logger.info("Computing composite toxicity score")
    logger.debug(f"Input values: {values}")
    
    for key, value in values.items():
        logger.debug(f"[SCORING] {key} = {value} ({type(value)})")

    try:
        score = sum(WEIGHTS[k] * values[k] for k in WEIGHTS)
    except KeyError as e:
        logger.error(f"Missing key in input values: {e}")
        raise

    final_score = round(min(max(score, 0), 1), 2)
    logger.debug(f"Final composite score: {final_score}")
    return final_score


def calculate_alert_penalty(alert_count: int):
    penalty = min(alert_count * 0.05, 1.0)
    logger.info(f"Calculating alert penalty for {alert_count} alerts: {penalty}")
    return penalty


def compute_accumulation_penalty(accumulation: dict):
    logger.info("Calculating tissue accumulation penalty")
    logger.debug(f"Tissue accumulation input: {accumulation}")

    penalty = 0
    mapping = {"low": 0, "moderate": 0.05, "high": 0.1}

    for organ, level in accumulation.items():
        organ_penalty = mapping.get(level.lower(), 0)
        logger.debug(f"{organ}: {level} → Penalty {organ_penalty}")
        penalty += organ_penalty

    final_penalty = round(penalty, 2)
    logger.debug(f"Total accumulation penalty: {final_penalty}")
    return final_penalty
