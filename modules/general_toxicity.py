import numpy as np
import joblib
import pandas as pd
from tqdm import tqdm
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from tdc.single_pred import Tox
from tdc.utils import retrieve_label_name_list
from modules.input_preprocessing import preprocess_smiles
from pathlib import Path
import os
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from utils.logger import logger

mlflow.set_tracking_uri("file://"+str(Path(__file__).resolve().parent.parent / "mlruns"))


def convert_smiles_column(X: pd.Series) -> pd.DataFrame:
    ecfp_list = []
    maccs_list = []

    for smiles in tqdm(X, desc="Processing SMILES"):
        processed = preprocess_smiles(smiles)
        ecfp_list.append(processed["ecfp"])
        maccs_list.append(processed["maccs"])

    logger.debug(f"Encoded {len(X)} SMILES entries into ECFP + MACCS features.")
    ecfp_array = np.vstack(ecfp_list)
    maccs_array = np.vstack(maccs_list)
    return pd.DataFrame(np.concatenate([ecfp_array, maccs_array], axis=1))


def train_ld50_model():
    with mlflow.start_run(run_name="LD50_Model_Training"):
        logger.info("Training LD50 regression model...")
        data = Tox(name='LD50_Zhu')
        tox = data.get_split(method='random')

        X_train, y_train = tox['train'].iloc[:, 1], tox['train'].iloc[:, 2]
        X_val, y_val = tox['valid'].iloc[:, 1], tox['valid'].iloc[:, 2]
        X_test, y_test = tox['test'].iloc[:, 1], tox['test'].iloc[:, 2]

        X_train_enc = convert_smiles_column(X_train)
        X_val_enc = convert_smiles_column(X_val)
        X_test_enc = convert_smiles_column(X_test)

        model = XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

        # Log hyperparameters
        mlflow.log_params(model.get_params())

        # Fit model
        model.fit(X_train_enc, y_train, eval_set=[(X_val_enc, y_val)], verbose=True)

        # Predict and evaluate
        y_pred = model.predict(X_test_enc)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("test_mse", mse)
        mlflow.log_metric("test_r2", r2)

        logger.info(f"LD50 MSE: {mse:.4f}")
        logger.info(f"LD50 R²: {r2:.4f}")

        # Log model with input schema
        X_test_enc = X_test_enc.astype(np.float64) 
        input_example = X_test_enc.iloc[:1]
        signature = infer_signature(X_test_enc, y_pred[:1])
        mlflow.sklearn.log_model(
            sk_model=model,
            name="ld50_model",
            input_example=input_example,
            signature=signature
        )

        # Save locally as well
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/ld50_xgb_model.pkl")
        logger.success("LD50 model saved to models/ld50_xgb_model.pkl")


def train_carcinogenicity_model():
    with mlflow.start_run(run_name="Carcinogenicity_Model_Training"):
        logger.info("Training Carcinogenicity classification model...")
        data = Tox(name='Carcinogens_Lagunin')
        carc = data.get_split(method='random')

        X_train, y_train = carc['train'].iloc[:, 1], carc['train'].iloc[:, 2]
        X_val, y_val = carc['valid'].iloc[:, 1], carc['valid'].iloc[:, 2]
        X_test, y_test = carc['test'].iloc[:, 1], carc['test'].iloc[:, 2]

        X_train_enc = convert_smiles_column(X_train)
        X_val_enc = convert_smiles_column(X_val)
        X_test_enc = convert_smiles_column(X_test)

        model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )

        mlflow.log_params(model.get_params())
        model.fit(X_train_enc, y_train, eval_set=[(X_val_enc, y_val)], verbose=True)

        y_pred = model.predict(X_test_enc)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        logger.info(f"Carcinogenicity Accuracy: {acc:.4f}")
        logger.info(f"Carcinogenicity F1 Score: {f1:.4f}")

        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_f1_score", f1)
        
        # Log model with input schema
        X_test_enc = X_test_enc.astype(np.float64) 
        input_example = X_test_enc.iloc[:1]
        signature = infer_signature(X_test_enc, y_pred[:1])
        mlflow.sklearn.log_model(
            sk_model=model,
            name="carcinogenicity_model",
            input_example=input_example,
            signature=signature
        )

        os.makedirs('models', exist_ok=True)
        joblib.dump(model, "models/carcinogenicity_xgb_model.pkl")
        logger.success("Carcinogenicity model saved to models/carcinogenicity_xgb_model.pkl")

def train_general_tox_model():
    with mlflow.start_run(run_name="GeneralTox_Model_Training"):
        logger.info("Training General Toxicity classifier (Tox21)...")
        label_list = retrieve_label_name_list('Tox21')
        data = Tox(name='Tox21', label_name=label_list[0])
        tox = data.get_split(method='random')

        X_train, y_train = tox['train'].iloc[:, 1], tox['train'].iloc[:, 2]
        X_val, y_val = tox['valid'].iloc[:, 1], tox['valid'].iloc[:, 2]
        X_test, y_test = tox['test'].iloc[:, 1], tox['test'].iloc[:, 2]

        X_train_enc = convert_smiles_column(X_train)
        X_val_enc = convert_smiles_column(X_val)
        X_test_enc = convert_smiles_column(X_test)

        model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )

        mlflow.log_params(model.get_params())
        model.fit(X_train_enc, y_train, eval_set=[(X_val_enc, y_val)], verbose=True)

        y_pred = model.predict(X_test_enc)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        logger.info(f"General Tox Accuracy: {acc:.4f}")
        logger.info(f"General Tox F1 Score: {f1:.4f}")

        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_f1_score", f1)
        
        # Log model with input schema
        X_test_enc = X_test_enc.astype(np.float64) 
        input_example = X_test_enc.iloc[:1]
        signature = infer_signature(X_test_enc, y_pred[:1])
        mlflow.sklearn.log_model(
            sk_model=model,
            name="general_tox_model",
            input_example=input_example,
            signature=signature
        )

        os.makedirs('models',exist_ok=True)
        joblib.dump(model, "models/generaltox_xgb_model.pkl")
        logger.success("General Tox model saved to models/generaltox_xgb_model.pkl")

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

    ld50_pred = float(ld50_model.predict(features)[0])
    carcinogenicity_pred = float(carcinogenicity_model.predict_proba(features)[0][1])
    general_tox_pred = float(generaltox_model.predict_proba(features)[0][1])

    logger.info(f"Predictions — LD50: {ld50_pred:.2f}, Carcinogenicity: {carcinogenicity_pred:.2f}, General Tox: {general_tox_pred:.2f}")

    return {
        "ld50": round(ld50_pred, 2),
        "carcinogenicity": round(carcinogenicity_pred, 2),
        "general_tox": round(general_tox_pred, 2)
    }

if __name__ == "__main__":
    train_ld50_model()
    train_carcinogenicity_model()
    train_general_tox_model()
