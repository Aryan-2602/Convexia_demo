from modules import structural_alerts

def test_structural_alerts_output_type():
    alerts = structural_alerts.check_structural_alerts("CCO")
    assert isinstance(alerts, list)
