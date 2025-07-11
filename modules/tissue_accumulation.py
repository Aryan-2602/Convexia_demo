import numpy as np
import joblib
import pandas as pd
from tqdm import tqdm
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from modules.input_preprocessing import preprocess_smiles
from utils.logger import logger
import os

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

# ---- TRAINING FUNCTIONS ---- #

def train_bbb_model():
    logger.info("Training BBB (Blood-Brain Barrier) model...")
    train = pd.read_csv("data/bbb_logbb/bbb_logbb_train.csv")
    val = pd.read_csv("data/bbb_logbb/bbb_logbb_val.csv")
    test = pd.read_csv("data/bbb_logbb/bbb_logbb_test.csv")

    X_train, y_train = train["smiles_standarized"], train["label"]
    X_val, y_val = val["smiles_standarized"], val["label"]
    X_test, y_test = test["smiles_standarized"], test["label"]

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
    model.fit(X_train_enc, y_train, eval_set=[(X_val_enc, y_val)], verbose=True)

    preds = model.predict(X_test_enc)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    logger.info(f"BBB Model Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    joblib.dump(model, "models/bbb_xgb_model.pkl")
    logger.success("BBB model saved to models/bbb_xgb_model.pkl")

def train_oct2_model():
    logger.info("Training OCT2 model (kidney accumulation)...")
    train = pd.read_csv("data/oct2/oct2_train.csv")
    val = pd.read_csv("data/oct2/oct2_val.csv")
    test = pd.read_csv("data/oct2/oct2_test.csv")

    X_train, y_train = train["smiles_standarized"], train["label"]
    X_val, y_val = val["smiles_standarized"], val["label"]
    X_test, y_test = test["smiles_standarized"], test["label"]

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
    model.fit(X_train_enc, y_train, eval_set=[(X_val_enc, y_val)], verbose=True)

    preds = model.predict(X_test_enc)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    logger.info(f"OCT2 Model Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    joblib.dump(model, "models/oct2_xgb_model.pkl")
    logger.success("OCT2 model saved to models/oct2_xgb_model.pkl")

def train_vd_model():
    logger.info("Training VD (Volume of Distribution) model...")
    train = pd.read_csv("data/vd/vd_train.csv")
    val = pd.read_csv("data/vd/vd_val.csv")
    test = pd.read_csv("data/vd/vd_test.csv")

    X_train, y_train = train["smiles_standarized"], train["label"]
    X_val, y_val = val["smiles_standarized"], val["label"]
    X_test, y_test = test["smiles_standarized"], test["label"]

    X_train_enc = convert_smiles_column(X_train)
    X_val_enc = convert_smiles_column(X_val)
    X_test_enc = convert_smiles_column(X_test)

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train_enc, y_train, eval_set=[(X_val_enc, y_val)], verbose=True)

    preds = model.predict(X_test_enc)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    logger.info(f"VD Model MSE: {mse:.4f}, R²: {r2:.4f}")
    joblib.dump(model, "models/vd_xgb_model.pkl")
    logger.success("VD model saved to models/vd_xgb_model.pkl")

# ---- INFERENCE FUNCTION FOR MAIN PIPELINE ---- #

def predict_tissue_accumulation(smiles: str):
    from pathlib import Path
    logger.info(f"Predicting tissue accumulation for: {smiles}")

    model_dir = Path("models")
    try:
        bbb_model = joblib.load(model_dir / "bbb_xgb_model.pkl")
        oct2_model = joblib.load(model_dir / "oct2_xgb_model.pkl")
        vd_model = joblib.load(model_dir / "vd_xgb_model.pkl")
    except FileNotFoundError as e:
        logger.error(f"Model loading failed: {e}")
        raise

    features = convert_smiles_column([smiles])

    bbb_prob = bbb_model.predict_proba(features)[0][1]
    oct2_prob = oct2_model.predict_proba(features)[0][1]
    vd_pred = vd_model.predict(features)[0]

    if vd_pred <= 0.3:
        vd_level = "low"
    elif vd_pred <= 0.55:
        vd_level = "moderate"
    else:
        vd_level = "high"

    result = {
        "brain": "high" if bbb_prob > 0.6 else "moderate" if bbb_prob > 0.4 else "low",
        "kidney": "high" if oct2_prob > 0.6 else "moderate" if oct2_prob > 0.4 else "low",
        "liver": vd_level
    }

    logger.debug(f"Tissue accumulation result: {result}")
    return result

# ---- OPTIONAL: RUN ALL TRAINERS ---- #
if __name__ == "__main__":
    train_bbb_model()
    train_oct2_model()
    train_vd_model()
