from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.tasks import run_pipeline_task
from celery.result import AsyncResult
from utils.logger import logger
from fastapi import FastAPI
from contextlib import asynccontextmanager
from scheduler import schedule_cleanup

@asynccontextmanager
async def lifespan(app:FastAPI):
    schedule_cleanup()

app = FastAPI(title="Toxicity Prediction API",lifespan=lifespan)

class SMILESInput(BaseModel):
    smiles: str
    profile: str

@app.post("/predict")
def submit_task(input: SMILESInput):
    task = run_pipeline_task.delay(input.smiles, input.profile)
    logger.info(f"Submitted task {task.id}")
    return {"job_id": task.id}


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
    

