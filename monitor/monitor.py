# monitor/monitor.py
import mlflow
from mlflow.tracking import MlflowClient
from rich import print
from datetime import datetime

def discover_and_monitor_models():
    client = MlflowClient()
    now = datetime.utcnow().isoformat()
    print(f"[bold green]Monitoring run started at {now}[/bold green]\n")

    for model in client.search_registered_models():
        model_name = model.name
        print(f"\n[bold cyan]Model:[/bold cyan] {model_name}")
        
        # Get latest version (staging or production)
        versions = model.latest_versions
        for version_info in versions:
            version = version_info.version
            status = version_info.current_stage
            run_id = version_info.run_id
            print(f"  - Version: {version} | Stage: {status}")

            # Fetch run metrics
            run = client.get_run(run_id)
            metrics = run.data.metrics
            print(f"    Metrics: {metrics}")

if __name__ == "__main__":
    discover_and_monitor_models()
