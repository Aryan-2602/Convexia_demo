import pandas as pd
from mordred import Calculator, descriptors
from rdkit import Chem
from rdkit.Chem import AllChem

def convert_smiles_column(smiles_list):
    """
    Takes a list of SMILES strings and returns a DataFrame of computed descriptors.
    Invalid SMILES will be skipped.
    """
    calc = Calculator(descriptors, ignore_3D=True)
    mols = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            mols.append(mol)
        else:
            raise ValueError(f"Invalid SMILES string: {smi}")

    df = calc.pandas(mols)
    df = df.replace([float("inf"), float("-inf")], pd.NA).dropna(axis=1)
    return df
