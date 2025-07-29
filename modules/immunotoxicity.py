from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
from utils.logger import logger
import torch
import mlflow
from pathlib import Path
import os

torch.set_num_threads(1)

logger.info("Loading the TxGemma 2B model for immunotoxicity...")

MODEL_VARIANT = "2b-predict"
MODEL_ID = f"google/txgemma-{MODEL_VARIANT}"
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# 4-bit quantization config for efficient inference

# Explicitly download the full snapshot locally
local_dir = snapshot_download(
    repo_id=MODEL_ID,
    revision="main",
    token=HF_TOKEN,
    local_dir="/tmp/txgemma-2b-predict",
    local_dir_use_symlinks=False
)

# Load model/tokenizer from the downloaded folder
tokenizer = AutoTokenizer.from_pretrained(local_dir)
model = AutoModelForCausalLM.from_pretrained(
    local_dir,
    device_map="auto",
    trust_remote_code=True,
)

# Set MLflow tracking
mlflow.set_tracking_uri("file://" + str(Path(__file__).resolve().parent.parent / "mlruns"))
mlflow.set_experiment("TxGemma_immunotoxicity")

# Define prompt template
def format_prompt(smiles: str) -> str:
    return (
        f"Given the molecular SMILES string below, predict the immunotoxicity score on a scale from 0 to 1.\n"
        f"SMILES: {smiles}\n"
        f"Immunotoxicity Score:"
    )

# Main function
def predict_immunotoxicity(smiles: str) -> float:
    logger.info(f"Predicting immunotoxicity using TxGemma for: {smiles}")
    prompt = format_prompt(smiles)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with mlflow.start_run(run_name="immunotoxicity_inference", nested=True):
        mlflow.set_tag("model", "txgemma-2b-immunotox")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.7,
                do_sample=False
            )

        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract score from generated text
        try:
            score_str = decoded_output.split("Immunotoxicity Score:")[-1].strip().split()[0]
            score = float(score_str)
            score = round(min(max(score, 0), 1), 4)
        except Exception as e:
            logger.error(f"Failed to parse immunotoxicity score: {decoded_output}")
            raise ValueError("Model output could not be parsed into a float.")

        mlflow.log_metric("immunotoxicity_score", score)
        return score
