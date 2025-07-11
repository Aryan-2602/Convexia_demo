import pytest
from modules import input_preprocessing

def test_preprocess_valid_smiles():
    smiles = "CCO"
    result = input_preprocessing.preprocess_smiles(smiles)
    assert "ecfp" in result and "maccs" in result
    assert result["ecfp"].shape[0] == 2048
