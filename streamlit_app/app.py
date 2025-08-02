import streamlit as st
import json
from pathlib import Path
from datetime import datetime
from rdkit import Chem
import os

# Add root directory to path to import modules
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from run_pipeline import run_toxicity_pipeline  # make sure it's the right import

st.set_page_config(page_title="Toxicity Predictor", layout="centered")
st.title("🧪 Toxicity Prediction for Small Molecules")

# Input
smiles = st.text_input("Enter SMILES string", placeholder="e.g., CCO")

# Submit
if st.button("Run Prediction"):
    if not smiles:
        st.warning("Please enter a valid SMILES string.")
    else:
        with st.spinner("🔬 Predicting..."):
            result = run_toxicity_pipeline(smiles)

        if "error" in result:
            st.error(f"❌ Prediction failed: {result['error']}")
        else:
            st.success("✅ Prediction complete!")

            # Save output JSON
            mol = Chem.MolFromSmiles(smiles)
            base_name = Chem.MolToSmiles(mol, canonical=True).replace('/', '_').replace('\\', '_') if mol else "invalid_smiles"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{base_name}_{timestamp}.json"

            output_dir = Path(__file__).resolve().parent.parent / "outputs"
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / filename

            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)

            st.markdown("### 📋 Output Summary")
            st.json(result)

            st.markdown(f"📂 Output saved to: `outputs/{filename}``")

            # SHAP Visualizations
            st.markdown("### 🔎 SHAP Explainability Plots")

            shap_dir = Path(__file__).resolve().parent.parent / "outputs" / "shap"

            shap_plots = {
                "General Toxicity (Tree-based)": shap_dir / "general_tox.png",
                "Tox21 Transformer (Transformer)": shap_dir / "tox21_transformer.png",
                "hERG Toxicity (TDC-style)": shap_dir / "herg_model.png"
            }

            for title, path in shap_plots.items():
                if path.exists():
                    st.subheader(f"📊 {title}")
                    st.image(str(path), use_column_width=True)
                else:
                    st.warning(f"SHAP plot for '{title}' not found.")
