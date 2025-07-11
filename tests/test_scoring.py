from modules import scoring

def test_score_with_sample_inputs():
    sample = {
        "general_tox": 0.5,
        "organ_tox_avg": 0.6,
        "neurotox": 0.7,
        "mito_tox": 0.4,
        "morpho_tox": 0.3,
        "accumulation_penalty": 0.2,
        "immunotox": 0.1,
        "structural_alert_penalty": 0.05
    }
    score = scoring.compute_composite_score(sample)
    assert 0 <= score <= 1
