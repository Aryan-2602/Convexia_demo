from rdkit import Chem
from rdkit.Chem import FilterCatalog
from utils.logger import logger

def check_structural_alerts(smiles: str):
    logger.info(f"Checking structural alerts for SMILES: {smiles}")
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.warning("Invalid SMILES string. Returning empty alert set.")
        return {"alerts": [], "alert_count": 0}

    # Combine PAINS and BRENK filters
    logger.debug("Initializing structural alert catalogs: PAINS + BRENK")
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    catalog = FilterCatalog.FilterCatalog(params)

    # Get triggered alerts
    alerts = catalog.GetMatches(mol)
    alert_names = [alert.GetDescription() for alert in alerts]

    logger.info(f"{len(alert_names)} structural alerts triggered.")
    if alert_names:
        logger.debug(f"Triggered alerts: {alert_names}")

    return {
        "alerts": alert_names,
        "alert_count": len(alert_names)
    }
