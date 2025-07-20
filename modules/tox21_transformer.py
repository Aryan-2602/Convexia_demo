
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
torch.set_num_threads(1)
import numpy as np

# Hugging Face model ID from the Hub
HUGGINGFACE_MODEL_ID = "mikemayuare/SELFY-BPE-tox21" # or any other compatible model
device = torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(HUGGINGFACE_MODEL_ID).to(device)

def predict_tox21_score(smiles: str) -> float:
    print(f"\n[INFO] Predicting Tox21 toxicity for: {smiles}")
    inputs = tokenizer(smiles, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    predicted_score = float(np.max(probs))
    predicted_label = float(np.argmax(probs))

    print(f"[RESULT] Predicted Score: {predicted_score:.4f}, Label: {predicted_label}")
    return round(predicted_score, 2)
