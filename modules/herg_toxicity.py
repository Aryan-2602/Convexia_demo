import json, sys, traceback, os
from tdc import tdc_hf_interface

MODEL_ID = "hERG_Karim-CNN"

def predict_herg_toxicity(smiles: str):
    tdc_hf = tdc_hf_interface(MODEL_ID)
    cache_dir = os.path.join(os.path.dirname(__file__), '..', '.tdc_cache')
    os.makedirs(cache_dir, exist_ok=True)
    dp_model = tdc_hf.load_deeppurpose(cache_dir)
    result = tdc_hf.predict_deeppurpose(dp_model, [smiles])
    return result

        
