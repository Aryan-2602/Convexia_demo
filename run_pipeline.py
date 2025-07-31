import json
import os
import sys
from pathlib import Path
from datetime import datetime

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
    scoring,
    tox21_transformer,
    ames_toxicity,
    herg_toxicity,
    metabolism
)

from rdkit import Chem
import mlflow

mlflow.set_tracking_uri("file://" + str(Path(__file__).resolve().parent.parent / "mlruns"))
mlflow.set_experiment("toxicity_pipeline")

def run_toxicity_pipeline(smiles: str):
    try:
        with mlflow.start_run(run_name="toxicity_session"):

            mlflow.log_param("smiles_input", smiles)

            # 1. Input Preprocessing
            mol_data = input_preprocessing.preprocess_smiles(smiles)

            # 2. Structural Alerts
            alert_data = structural_alerts.check_structural_alerts(smiles)
            alert_list = alert_data["alerts"]
            alert_count = alert_data["alert_count"]
            structural_alert_penalty = scoring.calculate_alert_penalty(alert_count)
            mlflow.log_metric("alert_count", alert_count)

            # 3. General Toxicity
            general = general_toxicity.predict_general_toxicity(smiles)
            mlflow.log_metric("ld50", general["ld50"])
            mlflow.log_metric("general_tox", general["general_tox"])
            mlflow.log_metric("carcinogenicity", general["carcinogenicity"])

            # 4. Organ Toxicity
            organ = organ_toxicity.predict_organ_toxicity(smiles)
            organ_tox_avg = sum(organ.values()) / 3
            mlflow.log_metric("organ_tox_avg", organ_tox_avg)

            # 5. Neurotoxicity
            neurotox = neurotoxicity.predict_neurotoxicity(smiles)
            mlflow.log_metric("neurotoxicity", neurotox)

            # 6. Mitochondrial Toxicity
            mito_tox = mito_toxicity.predict_mito_toxicity(smiles)
            mlflow.log_metric("mitochondrial_tox", mito_tox)

            # 7. Tissue Accumulation
            accumulation = tissue_accumulation.predict_tissue_accumulation(smiles)
            accumulation_penalty = scoring.compute_accumulation_penalty(accumulation)
            mlflow.log_metric("accumulation_penalty", accumulation_penalty)

            # 8. Morphological Cytotoxicity
            morpho_tox = morpho_cytotoxicity.predict_morphological_cytotoxicity(smiles)
            mlflow.log_metric("morphological_tox", morpho_tox)

            # 9. Immunotoxicity
            immunotox = immunotoxicity.predict_immunotoxicity(smiles)
            mlflow.log_metric("immunotoxicity", immunotox)

            # 10. Tox21 Transformer
            tox21_score = tox21_transformer.predict_tox21_score(smiles)
            mlflow.log_metric("tox21_score", tox21_score)

            # 11. Ames
            ames_score = ames_toxicity.predict_ames_toxicity(smiles)
            mlflow.log_metric("ames_score", ames_score)

            # 12. hERG
            herg_score = herg_toxicity.predict_herg_toxicity(smiles)
            mlflow.log_metric("herg_score", herg_score)
            
            # 13. Metabolism
            metabolism_score = metabolism.predict_metabolism(smiles)
            mlflow.log_metric("metabolism_score",metabolism_score)

            # 14. Explainability
            confidence = explainability.compute_confidence()
            mlflow.log_metric("model_confidence", confidence)

            disagreements = explainability.find_disagreements(organ)

            # 14. Composite Score
            values = {
                "general_tox": general["general_tox"],
                "organ_tox_avg": organ_tox_avg,
                "neurotox": neurotox,
                "mito_tox": mito_tox,
                "morpho_tox": morpho_tox,
                "accumulation_penalty": accumulation_penalty,
                "immunotox": immunotox,
                "structural_alert_penalty": structural_alert_penalty,
                "tox21_score": tox21_score,
                "ames_toxicity_score": ames_score,
                "herg_toxicity_score": herg_score,
                "metabolism_score":metabolism_score
            }

            composite_score = scoring.compute_composite_score(values)
            mlflow.log_metric("composite_score", composite_score)

            # Flags
            flags = []
            if organ_tox_avg > 0.4:
                flags.append("high organ-specific toxicity")
            if morpho_tox > 0.4:
                flags.append("morphological concern")
            if alert_count > 0:
                flags.append("structural alerts triggered")
            if tox21_score > 0.4:
                flags.append("tox transformer model indicates toxicity")
            if ames_score > 0.4:
                flags.append("ames model indicates toxicity")
            if herg_score < 0.001:
                flags.append("herg model indicates heart risk")
            if metabolism_score>0.4:
                flags.append("metabolism score indicates a metabolism risk")

            mlflow.log_text("\n".join(flags), artifact_file="flags.txt")

            # Output JSON
            output = {
                "composite_score": composite_score,
                "organ_toxicity": organ,
                "neurotoxicity": round(neurotox, 2),
                "mitochondrial_toxicity": round(mito_tox, 2),
                "tissue_accumulation": accumulation,
                "morphological_cytotoxicity": round(morpho_tox, 2),
                "immunotoxicity": round(immunotox, 2),
                "tox21_score": round(tox21_score, 2),
                "ames_toxicity_score": round(ames_score, 2),
                "herg_toxicity_score": round(herg_score, 2),
                "structural_alerts": alert_list,
                "ld50": general["ld50"],
                "metabolism_score":metabolism_score,
                "model_confidence": confidence,
                "flags": flags
            }

            return output

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    smiles_input = input("Enter SMILES string: ")
    result = run_toxicity_pipeline(smiles_input)
    print(json.dumps(result, indent=2))

    mol = Chem.MolFromSmiles(smiles_input)
    base_name = Chem.MolToSmiles(mol, canonical=True).replace('/', '_').replace('\\', '_') if mol else "invalid_smiles"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_name}_{timestamp}.json"

    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    mlflow.log_artifact(output_path)
    print(f"\n✅ Output saved to: outputs/{filename}")
