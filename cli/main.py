# cli/main.py
import click
import json
from run_pipeline import run_toxicity_pipeline

@click.group()
def cli():
    """Toxicity Prediction CLI"""
    pass

@cli.command()
@click.argument("smiles")
def predict(smiles):
    """Predict toxicity for a single SMILES string."""
    result = run_toxicity_pipeline(smiles)
    click.echo(json.dumps(result, indent=2))

@cli.command()
@click.argument("smi_file", type=click.Path(exists=True))
@click.option("--out", "-o", type=click.Path(), default="results.json", help="Output file to save results")
def batch(smi_file, out):
    """Run batch prediction from a .smi file"""
    results = {}
    with open(smi_file, "r") as f:
        for line in f:
            smiles = line.strip()
            if smiles:
                result = run_toxicity_pipeline(smiles)
                results[smiles] = result

    with open(out, "w") as fout:
        json.dump(results, fout, indent=2)

    click.echo(f"✅ Batch predictions saved to {out}")
