import json
from modules import (
    input_preprocessing,
    structural_alerts,
    general_toxicity,
    organ_toxicity,
    neurotoxicity,
    mito_toxicity,
    tissue_accumulation,
    morpho_cytotoxicity,
    immunotoxicity,
    explainability,
    scoring
)

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rdkit import Chem
from datetime import datetime


def run_toxicity_pipeline(smiles: str):
    try:
        # 1. Input Preprocessing
        mol_data = input_preprocessing.preprocess_smiles(smiles)

        # 2. Structural Alerts
        alert_data = structural_alerts.check_structural_alerts(smiles)  # returns dict
        alert_list = alert_data["alerts"]
        alert_count = alert_data["alert_count"]
        structural_alert_penalty = scoring.calculate_alert_penalty(alert_count)

        # 3. General Toxicity
        general = general_toxicity.predict_general_toxicity(smiles)

        # 4. Organ Toxicity
        organ = organ_toxicity.predict_organ_toxicity(smiles)

        if not isinstance(organ, dict):
             raise TypeError(f"Expected dict from predict_organ_toxicity but got {type(organ).__name__}: {organ}")

        organ_tox_avg = sum(organ.values()) / 3

        # 5. Neurotoxicity
        neurotox = neurotoxicity.predict_neurotoxicity(smiles)

        # 6. Mitochondrial Toxicity
        mito_tox = mito_toxicity.predict_mito_toxicity(smiles)

        # 7. Tissue Accumulation
        accumulation = tissue_accumulation.predict_tissue_accumulation(smiles)
        accumulation_penalty = scoring.compute_accumulation_penalty(accumulation)

        # 8. Morphological Cytotoxicity
        morpho_tox = morpho_cytotoxicity.predict_morphological_cytotoxicity(smiles)

        # 9. Immunotoxicity
        immunotox = immunotoxicity.predict_immunotoxicity(smiles)

        # 10. Explainability
        confidence = explainability.compute_confidence()
        disagreements = explainability.find_disagreements(organ)

        # 11. Scoring
        composite_score = scoring.compute_composite_score({
            "general_tox": general["general_tox"],
            "organ_tox_avg": organ_tox_avg,
            "neurotox": neurotox,
            "mito_tox": mito_tox,
            "morpho_tox": morpho_tox,
            "accumulation_penalty": accumulation_penalty,
            "immunotox": immunotox,
            "structural_alert_penalty": structural_alert_penalty
        })

        # Flags
        flags = []
        if any(score > 0.8 for score in organ.values()):
            flags.append("high organ-specific toxicity")
        if morpho_tox > 0.8:
            flags.append("morphological concern")
        if alert_count > 0:
            flags.append("structural alerts triggered")

        # Output JSON
        output = {
            "composite_score": composite_score,
            "organ_toxicity": organ,
            "neurotoxicity": round(neurotox, 2),
            "mitochondrial_toxicity": round(mito_tox, 2),
            "tissue_accumulation": accumulation,
            "morphological_cytotoxicity": round(morpho_tox, 2),
            "immunotoxicity": round(immunotox, 2),
            "structural_alerts": alert_list,
            "ld50": general["ld50"],
            "flags": flags,
            "model_confidence": confidence
        }

        return output

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    from rdkit import Chem
    from datetime import datetime

    smiles_input = input("Enter SMILES string: ")
    result = run_toxicity_pipeline(smiles_input)

    # Print the result nicely
    print(json.dumps(result, indent=2))

    # Generate safe name from SMILES (canonicalized)
    mol = Chem.MolFromSmiles(smiles_input)
    if mol:
        base_name = Chem.MolToSmiles(mol, canonical=True).replace('/', '_').replace('\\', '_')
    else:
        base_name = "invalid_smiles"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_name}_{timestamp}.json"

    # Ensure outputs folder exists
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Save output
    with open(os.path.join(output_dir, filename), "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n✅ Output saved to: outputs/{filename}")
