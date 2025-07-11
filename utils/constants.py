# Scoring weights
SCORING_WEIGHTS = {
    "general_tox": 0.15,
    "organ_tox_avg": 0.2,
    "neurotox": 0.15,
    "mito_tox": 0.1,
    "morpho_tox": 0.1,
    "accumulation_penalty": 0.1,
    "immunotox": 0.1,
    "structural_alert_penalty": 0.1
}

# Penalty mapping for tissue accumulation levels
ACCUMULATION_PENALTY = {
    "low": 0.0,
    "moderate": 0.05,
    "high": 0.1
}

# Alert penalty per hit
ALERT_PENALTY_PER_ALERT = 0.05
MAX_ALERT_PENALTY = 1.0
