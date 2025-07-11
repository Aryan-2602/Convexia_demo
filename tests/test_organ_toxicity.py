from modules import organ_toxicity

def test_organ_toxicity_values():
    result = organ_toxicity.predict_organ_toxicity("CCO")
    assert all(0 <= v <= 1 for v in result.values())
