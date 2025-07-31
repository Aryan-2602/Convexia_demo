import streamlit as st
import json
from pathlib import Path
from datetime import datetime
from rdkit import Chem

# Add root directory to path to import modules
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import run_pipeline  # assumes the main pipeline is in main.py

st.set_page_config(page_title="Toxicity Predictor", layout="centered")
st.title("🧪 Toxicity Prediction for Small Molecules")

# Input
smiles = st.text_input("Enter SMILES string", placeholder="e.g., CCO")

# Submit
if st.button("Run Prediction"):
    if not smiles:
        st.warning("Please enter a valid SMILES string.")
    else:
        with st.spinner("Predicting..."):
            result = run_pipeline(smiles)

        if "error" in result:
            st.error(f"Prediction failed: {result['error']}")
        else:
            st.success("✅ Prediction complete!")

            # Save output JSON to disk
            mol = Chem.MolFromSmiles(smiles)
            base_name = Chem.MolToSmiles(mol, canonical=True).replace('/', '_').replace('\\', '_') if mol else "invalid_smiles"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{base_name}_{timestamp}.json"

            output_dir = Path(__file__).resolve().parent.parent / "outputs"
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / filename

            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)

            st.markdown("### Output Summary")
            st.json(result)

            st.markdown(f"📂 Output saved to: `outputs/{filename}`")
