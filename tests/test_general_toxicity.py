from modules import general_toxicity

def test_general_toxicity_keys():
    result = general_toxicity.predict_general_toxicity("CCO")
    assert all(key in result for key in ["ld50", "carcinogenicity", "general_tox"])
