import pytest
from modules import metabolism

@pytest.mark.unit
def test_metabolism_score_range():
    """
    Ensure that the metabolism module returns a score between 0 and 1.
    """
    smiles = "CCO"  # Ethanol
    score = metabolism.predict_metabolism(smiles)

    assert isinstance(score, float), "Output should be a float"
    assert 0.0 <= score <= 1.0, f"Score {score} should be in [0.0, 1.0]"
