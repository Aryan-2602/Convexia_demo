import pytest
from modules import immunotoxicity
import os


@pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip long-running test in CI")
@pytest.mark.unit
def test_immunotoxicity_prediction_score_range():
    """
    Ensure that the immunotoxicity model returns a float between 0 and 1.
    """
    smiles = "CCO"  # Ethanol
    score = immunotoxicity.predict_immunotoxicity(smiles)

    assert isinstance(score, float), "Output should be a float"
    assert 0.0 <= score <= 1.0, f"Score {score} should be in [0.0, 1.0]"
