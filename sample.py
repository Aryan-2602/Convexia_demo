from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import time
import os

torch.set_num_threads(1)

# Paths
LOCAL_MODEL_PATH = "models/tox_21_transformer"
device = torch.device("cpu")

# Load tokenizer and model
print("[DEBUG] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)

print("[DEBUG] Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    LOCAL_MODEL_PATH,
    local_files_only=True
).to(device)
print("[DEBUG] Model loaded successfully.")

# Sample SMILES input
test_smiles = "CCO"

# Tokenize input
print(f"\n[INFO] Tokenizing SMILES: {test_smiles}")
inputs = tokenizer(test_smiles, return_tensors="pt", padding=True, truncation=True)
print(f"[DEBUG] Tokenizer output type: {type(inputs)}")
print(f"[DEBUG] Tokenizer keys: {list(inputs.keys())}")
print(f"[DEBUG] input_ids shape: {inputs['input_ids'].shape}")

inputs = {k: v.to(device) for k, v in inputs.items()}

# Inference
with torch.no_grad():
    print("[DEBUG] Running model inference...")
    start = time.time()
    outputs = model(**inputs)
    end = time.time()
    print(f"[DEBUG] Inference time: {end - start:.2f} seconds")
    print(f"[DEBUG] outputs type: {type(outputs)}")
    print(f"[DEBUG] outputs keys: {outputs.keys()}")
    print(f"[DEBUG] logits shape: {outputs.logits.shape}")
    print(f"[DEBUG] logits tensor: {outputs.logits}")

# Softmax
logits = outputs.logits
probs = torch.softmax(logits, dim=1)
print(f"[DEBUG] softmax probs: {probs}")
print(f"[DEBUG] probs numpy: {probs.cpu().numpy()}")
print(f"[DEBUG] probs type: {type(probs)}, shape: {probs.shape}")

# Predictions
predicted_label = int(torch.argmax(probs, dim=1))
predicted_score = float(torch.max(probs))
print(f"\n[RESULT] Predicted Label: {predicted_label} (type: {type(predicted_label)})")
print(f"[RESULT] Predicted Score: {predicted_score:.4f} (type: {type(predicted_score)})")
