# modules/general_toxicity.py

import numpy as np
import joblib
import pandas as pd
from tqdm import tqdm
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from tdc.single_pred import Tox
from tdc.utils import retrieve_label_name_list
from modules.input_preprocessing import preprocess_smiles
from utils.logger import logger
from utils.explainability import explain_tree_model, compute_confidence, find_disagreements
from pathlib import Path
import os
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

mlflow.set_tracking_uri("file://" + str(Path(__file__).resolve().parent.parent / "mlruns"))

def convert_smiles_column(X: pd.Series) -> pd.DataFrame:
    ecfp_list = []
    maccs_list = []

    for smiles in tqdm(X, desc="Processing SMILES"):
        processed = preprocess_smiles(smiles)
        ecfp_list.append(processed["ecfp"])
        maccs_list.append(processed["maccs"])

    ecfp_array = np.vstack(ecfp_list)
    maccs_array = np.vstack(maccs_list)
    return pd.DataFrame(np.concatenate([ecfp_array, maccs_array], axis=1))


def predict_general_toxicity(smiles: str):
    logger.info(f"Running general toxicity prediction for: {smiles}")
    model_dir = Path("models")
    if not model_dir.exists():
        logger.error("Trained model directory not found.")
        raise FileNotFoundError("Trained model directory not found. Please train the models first.")

    ld50_model = joblib.load(model_dir / "ld50_xgb_model.pkl")
    carcinogenicity_model = joblib.load(model_dir / "carcinogenicity_xgb_model.pkl")
    generaltox_model = joblib.load(model_dir / "generaltox_xgb_model.pkl")
    logger.debug("Models loaded successfully.")

    features = convert_smiles_column([smiles])
    logger.debug("SMILES features encoded for prediction.")

    with mlflow.start_run(run_name="general_toxicity_inference", nested=False):
        mlflow.set_tag("module", "general_toxicity")
        mlflow.set_tag("task", "toxicity prediction")
        mlflow.set_tag("input_type", "SMILES")
        mlflow.set_tag("models_used", "LD50, Carcinogenicity, GeneralTox")

        ld50_pred = float(ld50_model.predict(features)[0])
        carcinogenicity_pred = float(carcinogenicity_model.predict_proba(features)[0][1])
        general_tox_pred = float(generaltox_model.predict_proba(features)[0][1])

        confidence = compute_confidence()
        mlflow.log_metric("confidence", confidence)

        mlflow.log_metric("ld50", ld50_pred)
        mlflow.log_metric("carcinogenicity", carcinogenicity_pred)
        mlflow.log_metric("general_tox", general_tox_pred)

        disagreements = find_disagreements({
            "ld50": ld50_pred,
            "carcinogenicity": carcinogenicity_pred,
            "general_tox": general_tox_pred
        })
        mlflow.set_tag("disagreements", str(len(disagreements)))

        # SHAP explanations
        explain_tree_model(ld50_model, features, output_path="outputs/shap/ld50_shap.png")
        explain_tree_model(carcinogenicity_model, features, output_path="outputs/shap/carcinogenicity_shap.png")
        explain_tree_model(generaltox_model, features, output_path="outputs/shap/generaltox_shap.png")

        logger.info(f"Predictions — LD50: {ld50_pred:.2f}, Carcinogenicity: {carcinogenicity_pred:.2f}, General Tox: {general_tox_pred:.2f}")

        return {
            "ld50": round(ld50_pred, 2),
            "carcinogenicity": round(carcinogenicity_pred, 2),
            "general_tox": round(general_tox_pred, 2),
            "confidence": confidence,
            "disagreements": disagreements
        }
