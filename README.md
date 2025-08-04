# 🧪 Toxicity & Safety Evaluation Pipeline

A modular pipeline for predicting toxicity and safety of small-molecule compounds from their SMILES strings. This system combines predictive models, structural alert filters, accumulation logic, and scoring heuristics to generate a comprehensive toxicity profile in JSON format.

## 🚀 Features

- ✅ Accepts SMILES strings as input  
- 🧠 Predicts:
  - General Toxicity: LD50, Carcinogenicity, Tox21
  - Organ Toxicity: Cardio, Hepatic, Renal
  - Neurotoxicity
  - Mitochondrial Toxicity
  - Morphological Cytotoxicity
  - Immunotoxicity
  - Tissue Accumulation (brain, kidney, liver)
- 🧨 Detects Structural Alerts (PAINS + BRENK)
- 📊 Composite Score based on weighted subcomponents
- 🔍 Explainability:
  - Model confidence
  - Disagreement among organ predictors
  - SHAP plots for XGBoost-based predictors
- 📈 MLflow tracking for:
  - Training runs
  - Inference predictions
  - SHAP visualizations
  - Tags and artifacts
- 🪵 Clean logging using Loguru
- 📁 Stores output as unique JSON files in `outputs/`

## 📁 Directory Structure

convexia_demo/
├── run_pipeline.py # Main pipeline entry point
├── outputs/ # Stores JSON results and SHAP plots
├── logs/ # Logging directory
├── models/ # Trained model files
├── data/ # Raw training datasets
│
├── utils/
│ ├── logger.py # Loguru logger configuration
│ └── explainability.py # SHAP + confidence + disagreement utilities
│
├── modules/
│ ├── input_preprocessing.py # SMILES featurization (ECFP + MACCS)
│ ├── general_toxicity.py # LD50, carcinogenicity, general toxicity models
│ ├── organ_toxicity.py # Cardio, hepatic, and renal toxicity predictors
│ ├── neurotoxicity.py # Neurotoxicity prediction (stub)
│ ├── mito_toxicity.py # Mitochondrial toxicity prediction (stub)
│ ├── morpho_cytotoxicity.py # Morphological cytotoxicity prediction (stub)
│ ├── immunotoxicity.py # Immunotoxicity prediction (stub)
│ ├── tissue_accumulation.py # BBB, OCT2, VD prediction + SHAP + logging
│ ├── structural_alerts.py # PAINS + BRENK alerts
│ ├── scoring.py # Composite score + penalties
│ └── register_models.py # Optional MLflow model registration helper

## 🧪 How to Run

```bash
# Step 1: Activate your environment
conda activate toxicity_classifier

# Step 2: Install the requirements
pip install -r requirements.txt

# Step 3: Run the pipeline
python run_pipeline.py
You will be prompted to enter a SMILES string. The system will run all modules and save the result as a JSON file in outputs/, while also logging metrics to MLflow and saving SHAP plots. 
```

🔬 Sample Output (Truncated)
{
  "composite_score": 0.36,
  "organ_toxicity": {
    "cardiotoxicity": 0.45,
    "hepatotoxicity": 0.62,
    "nephrotoxicity": 0.71
  },
  "neurotoxicity": 0.42,
  "mitochondrial_toxicity": 0.35,
  "tissue_accumulation": {
    "brain": "moderate",
    "kidney": "high",
    "liver": "high"
  },
  "morphological_cytotoxicity": 0.67,
  "immunotoxicity": 0.29,
  "structural_alerts": ["PAINS_A", "BRENK_3"],
  "ld50": 1.48,
  "flags": [
    "high organ-specific toxicity",
    "structural alerts triggered"
  ],
  "model_confidence": 0.93
}
⚙️ Composite Score Logic
The final score is calculated as a weighted sum of prediction components:
15% General Toxicity
20% Organ Toxicity (average)
15% Neurotoxicity
10% Mitochondrial Toxicity
10% Morphological Cytotoxicity
10% Immunotoxicity
10% Tissue Accumulation Penalty
10% Structural Alerts Penalty
Final score is clipped between 0 and 1 and rounded to two decimals.
📈 MLflow Tracking
All training and inference runs are tracked via MLflow:
Parameters, metrics, and tags
SHAP visualizations
Confidence and disagreement scores
Model artifacts (XGBoost binaries, input schema)
Tracked locally in the mlruns/ directory.
To launch the tracking UI:

mlflow ui
📌 Notes
Logging is enabled both to console and logs/pipeline.log
SHAP plots are saved in outputs/shap/
JSON filenames are uniquely generated using the compound's canonical SMILES
Stub models (e.g., mito, neuro, morpho) can be replaced by trained ones
To retrain general toxicity models:
python modules/general_toxicity.py
👩‍🔬 Acknowledgements
TDC for access to toxicity datasets
RDKit for molecular featurization and alerting
XGBoost for all classifier/regressor models
MLflow for experiment tracking
Loguru for structured logging
SHAP for explainability
🧠 TODOs
Replace stubbed models (mito, neuro, morpho, immune) with trained versions
Add uncertainty quantification for all predictors
Incorporate graph-based models and image-based inference where applicable
Add Streamlit/Gradio frontend for live demos
📬 Contact
For issues, contributions, or collaborations, feel free to reach out!
