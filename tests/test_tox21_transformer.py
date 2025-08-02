import pytest
from modules import tox21_transformer

@pytest.mark.unit
def test_tox21_score_prediction():
    """
    Ensure that the Tox21 transformer model returns a float between 0 and 1.
    """
    smiles = "CCO"  # Ethanol
    score = tox21_transformer.predict_tox21_score(smiles)

    assert isinstance(score, float), "Output should be a float"
    assert 0.0 <= score <= 1.0, f"Tox21 score {score} should be in [0.0, 1.0]"
