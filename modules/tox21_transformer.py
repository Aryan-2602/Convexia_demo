from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import mlflow
from pathlib import Path

# Hugging Face model ID from the Hub
HUGGINGFACE_MODEL_ID = "mikemayuare/SELFY-BPE-tox21"
device = torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(HUGGINGFACE_MODEL_ID).to(device)

mlflow.set_tracking_uri("file://" + str(Path(__file__).resolve().parent.parent / "mlruns"))
mlflow.set_experiment("tox21_transformer")

def predict_tox21_score(smiles: str) -> float:
    print(f"\n[INFO] Predicting Tox21 toxicity for: {smiles}")
    with mlflow.start_run(run_name="tox21_model", nested=True):
        mlflow.set_tag("model", "tox21_transformer")
        inputs = tokenizer(smiles, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        predicted_score = float(np.max(probs))
        predicted_label = float(np.argmax(probs))

        mlflow.log_param("model_name", HUGGINGFACE_MODEL_ID)
        mlflow.log_metric("tox21_score", predicted_score)
        mlflow.log_metric("predicted_label", predicted_label)

        print(f"[RESULT] Predicted Score: {predicted_score:.4f}, Label: {predicted_label}")
        return round(predicted_score, 2)
