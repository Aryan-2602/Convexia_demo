import pytest
from modules import ames_toxicity
import os

@pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI")
@pytest.mark.unit
def test_ames_toxicity_prediction_range():
    """
    Ensure that the Ames toxicity model returns a float between 0 and 1.
    """
    smiles = "CCO"  # Ethanol
    score = ames_toxicity.predict_ames_toxicity(smiles)

    assert isinstance(score, float), "Prediction should be a float"
    assert 0.0 <= score <= 1.0, f"Score {score} should be in [0.0, 1.0]"
