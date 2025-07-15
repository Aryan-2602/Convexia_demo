from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.tasks import run_pipeline_task
from celery.result import AsyncResult
from utils.logger import logger

app = FastAPI(title="Toxicity Prediction API")

class SMILESInput(BaseModel):
    smiles: str

@app.post("/predict")
def submit_task(input: SMILESInput):
    task = run_pipeline_task.delay(input.smiles)
    logger.info(f"Submitted task {task.id}")
    return {"task_id": task.id}

@app.get("/results/{task_id}")
def get_task_result(task_id: str):
    result = AsyncResult(task_id)
    if result.state == "PENDING":
        return {"status": "pending"}
    elif result.state == "SUCCESS":
        return {"status": "completed", "result": result.result}
    elif result.state == "FAILURE":
        logger.error(f"Task {task_id} failed: {result.result}")
        raise HTTPException(status_code=500, detail="Task failed")
    else:
        return {"status": result.state}
