from mlflow.tracking import MlflowClient
client = MlflowClient()

registered_models = client.search_registered_models()
for model in registered_models:
        print(f"Model Name: {model.name}")
