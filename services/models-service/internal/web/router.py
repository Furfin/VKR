from fastapi import APIRouter, Request, status
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pkg.minio.client import MinioClient
import pkg.storage.models.model as storage
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from internal.models.model import *
from fastapi.templating import Jinja2Templates
from typing import Dict, Any, List
import os
import uuid
from typing import Optional
import asyncio

minio_client = MinioClient(os.getenv("MINIO_ENDPOINT"),os.getenv("MINIO_ACCESS_KEY"),os.getenv("MINIO_SECRET_KEY"),False)

db_client = storage.ModelsStorageClient(os.getenv('DATABASE_URL'))

processing_tasks: Dict[str, asyncio.Task] = {}
target_processing_tasks: Dict[str, asyncio.Task] = {}
model_statuses: Dict[str, str] = {}


router = APIRouter()

@router.post("/api/CreateModel")
async def create_model(model_name: str):
    try:
        Model(model_name, minio_client, db_client)
        return {"msg": "ok"}
    except Exception as e:
        return {"Error": str(e)}

@router.get("/api/ListModel")
async def list_models():
    try:
        return db_client.list_objects()
    except Exception as e:
        return {"Error": str(e)}

@router.post("/api/TrainModel")
async def train_model(model_name: str, filename: str, num_epoch:int = 100):
    model = Model(model_name, minio_client, db_client)
    
    if filename in processing_tasks and not processing_tasks[filename].done():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Dataset is already being processed"
        )
    
    processing_task = asyncio.create_task(
        model.train_model_with(
            filename=filename,
            num_epoch=num_epoch
        )
    )

    processing_tasks[filename] = processing_task
    model_statuses[model_name] = "training"
    
    return JSONResponse(
        content={
            "message": "Training started",
        },
        status_code=status.HTTP_202_ACCEPTED
    )

@router.get("/api/TrainStatus")
async def get_training_status(model_name: str):
    if model_name not in model_statuses:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    if model_statuses[model_name] == "training":
        status_info = {
            "model_name": model_name,
            "status": model_statuses[model_name]
        }
        
        return status_info
    elif model_statuses[model_name] == "done":
        model = Model(model_name, minio_client, db_client)
        return {model.model_data["last_result"]}
        

@router.get("/")
async def home(request: Request):
    return {"msg": "models-service"}
