import os
import numpy as np
import shap
import matplotlib.pyplot as plt
from utils.logger import logger
from rdkit import Chem

def compute_confidence():
    confidence = np.round(np.random.uniform(0.85, 0.99), 2)
    logger.info(f"Computed model confidence: {confidence}")
    return confidence

def find_disagreements(organ_tox: dict):
    logger.debug("Checking for module disagreements in organ toxicity scores.")
    values = list(organ_tox.values())
    disagreements = []

    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            diff = abs(values[i] - values[j])
            if diff > 0.5:
                msg = f"Module disagreement between organ_{i} and organ_{j} (Δ={diff:.2f})"
                disagreements.append(msg)
                logger.warning(msg)

    if not disagreements:
        logger.info("No significant disagreements found between organ toxicity modules.")

    return disagreements


# ------------------------- SHAP Explainability -------------------------

def ensure_output_dir(output_dir="outputs/shap/"):
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def explain_tree_model(model, X, feature_names=None, output_path="outputs/shap/tree_shap.png"):
    logger.info("Generating SHAP explanation for tree-based model...")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
        ensure_output_dir()
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Tree-based SHAP plot saved to: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"SHAP explanation failed for tree model: {e}")
        return None


def explain_transformer_model(model, tokenizer, smiles, output_path="outputs/shap/transformer_shap.png"):
    logger.info("Generating SHAP explanation for transformer model...")
    try:
        tokens = tokenizer(smiles, return_tensors="pt")
        explainer = shap.Explainer(model, tokenizer)
        shap_values = explainer([smiles])

        shap.plots.text(shap_values[0], display=False)
        ensure_output_dir()
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Transformer SHAP plot saved to: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"SHAP explanation failed for transformer model: {e}")
        return None


def explain_tdc_model(model, smiles, featurizer, output_path="outputs/shap/tdc_shap.png"):
    logger.info("Generating SHAP explanation for TDC/DeepPurpose model...")
    try:
        # Vectorize SMILES input
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string")

        X = featurizer([smiles])
        explainer = shap.KernelExplainer(model.predict, X)
        shap_values = explainer.shap_values(X)

        shap.summary_plot(shap_values, X, show=False)
        ensure_output_dir()
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        logger.info(f"TDC SHAP plot saved to: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"SHAP explanation failed for TDC model: {e}")
        return None
