import pytest
from modules import herg_toxicity

@pytest.mark.unit
def test_herg_toxicity_prediction_range():
    """
    Ensure that the hERG toxicity module returns a float in [0.0, 1.0].
    """
    smiles = "CCO"  # Ethanol
    score = herg_toxicity.predict_herg_toxicity(smiles)

    assert isinstance(score, float), "Prediction should be a float"
    assert 0.0 <= score <= 1.0, f"Prediction score {score} should be in [0.0, 1.0]"
