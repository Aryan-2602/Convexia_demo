import joblib
import os
from utils.preprocessor import convert_smiles_column

def run_all_models(smiles: str):
    """
    Run prediction using all trained models on the given SMILES string.
    """
    input_df = convert_smiles_column([smiles])
    predictions = {}

    model_paths = {
        "ld50": "models/ld50_xgb_model.pkl",
        "carcinogenicity": "models/carcinogenicity_xgb_model.pkl",
        "generaltox": "models/generaltox_xgb_model.pkl",
        "bbb": "models/bbb_xgb_model.pkl",
        "oct2": "models/oct2_xgb_model.pkl",
        "vd": "models/vd_xgb_model.pkl",
    }

    for model_name, path in model_paths.items():
        if not os.path.exists(path):
            predictions[model_name] = "Model file not found"
            continue

        model = joblib.load(path)
        pred = model.predict(input_df)[0]
        predictions[model_name] = round(float(pred), 4)

    return predictions
