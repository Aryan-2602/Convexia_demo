import pytest
import json
from pathlib import Path

from run_pipeline import run_toxicity_pipeline

@pytest.mark.integration
def test_pipeline_output_structure():
    """
    Test the full toxicity pipeline for expected output structure and value ranges.
    """
    smiles = "CCO"  # Ethanol

    result = run_toxicity_pipeline(smiles)

    # Check no errors
    assert "error" not in result, f"Pipeline raised error: {result.get('error')}"

    # Check presence of core keys
    expected_keys = [
        "composite_score", "organ_toxicity", "neurotoxicity", "mitochondrial_toxicity",
        "tissue_accumulation", "morphological_cytotoxicity", "immunotoxicity",
        "tox21_score", "ames_toxicity_score", "herg_toxicity_score", "structural_alerts",
        "ld50", "metabolism_score", "model_confidence", "flags"
    ]
    for key in expected_keys:
        assert key in result, f"Missing key in result: {key}"

    # Check numerical score ranges
    assert 0.0 <= result["composite_score"] <= 1.0, "Composite score out of range"
    assert 0.0 <= result["tox21_score"] <= 1.0, "Tox21 score out of range"
    assert 0.0 <= result["ames_toxicity_score"] <= 1.0, "Ames score out of range"
    assert 0.0 <= result["herg_toxicity_score"] <= 1.0, "hERG score out of range"
    assert 0.0 <= result["metabolism_score"] <= 1.0, "Metabolism score out of range"
    assert 0.0 <= result["model_confidence"] <= 1.0, "Model confidence out of range"

    # Check structure of organ_toxicity and tissue_accumulation
    for organ in ["liver", "kidney", "heart"]:
        assert organ in result["organ_toxicity"], f"Missing organ: {organ}"

    for tissue in ["brain", "kidney", "liver"]:
        assert tissue in result["tissue_accumulation"], f"Missing tissue accumulation entry: {tissue}"

    # Check that flags is a list
    assert isinstance(result["flags"], list), "Flags should be a list"

    print("✅ Pipeline test passed for SMILES:", smiles)
