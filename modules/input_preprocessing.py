import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem import rdMolDescriptors
import numpy as np
from utils.logger import logger

def preprocess_smiles(smiles: str):
    logger.info(f"Preprocessing SMILES: {smiles}")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.error("Invalid SMILES string provided.")
        raise ValueError("Invalid SMILES string")

    mol = Chem.AddHs(mol)

    try:
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        logger.debug("3D embedding successful.")
    except Exception as e:
        logger.warning(f"3D embedding failed for {smiles}: {e}")

    ecfp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    maccs = MACCSkeys.GenMACCSKeys(mol)

    logger.debug(f"ECFP and MACCS fingerprints generated for {smiles}")

    return {
        "rdkit_mol": mol,
        "ecfp": np.array(ecfp),
        "maccs": np.array(maccs)
    }
