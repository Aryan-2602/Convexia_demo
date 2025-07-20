from utils.logger import logger

def compute_composite_score(values: dict):
    logger.info("Computing composite toxicity score")
    logger.debug(f"Input values: {values}")
    
    for key, value in values.items():
        logger.debug(f"[SCORING] {key} = {value} ({type(value)})")

    score = (
        0.15 * values["general_tox"] +
        0.2 * values["organ_tox_avg"] +
        0.15 * values["neurotox"] +
        0.1 * values["mito_tox"] +
        0.1 * values["morpho_tox"] +
        0.1 * values["accumulation_penalty"] +
        0.1 * values["immunotox"] +
        0.1 * values["structural_alert_penalty"]+
        0.2 * values["tox21_score"]+
        0.2 * values["ames_toxicity_score"]+
        100 * values['herg_toxicity_score']
    )
    

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
