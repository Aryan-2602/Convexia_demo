import pytest
from modules import tissue_accumulation

@pytest.mark.unit
def test_tissue_accumulation_prediction_structure():
    """
    Ensure that the tissue accumulation model returns a dictionary
    with valid organ keys and expected levels.
    """
    smiles = "CCO"  # Ethanol
    result = tissue_accumulation.predict_tissue_accumulation(smiles)

    # Validate structure
    assert isinstance(result, dict), "Result should be a dictionary"
    assert all(k in result for k in ["brain", "kidney", "liver"]), "Missing expected organ keys"

    # Validate content
    valid_levels = {"low", "moderate", "high"}
    for organ, level in result.items():
        assert level in valid_levels, f"Invalid level '{level}' for organ '{organ}'"
