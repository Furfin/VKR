from fastapi import APIRouter, Request, status, BackgroundTasks
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
async def train_model( background_tasks: BackgroundTasks, model_name: str, filename: str, num_epoch:int = 100):
    try:
        model = Model(model_name, minio_client, db_client)
        
        background_tasks.add_task(model.train_model_with,filename=filename,num_epoch=num_epoch)

        model_statuses[model_name] = "training"
        
        return JSONResponse(
            content={
                "message": "Training started",
            },
            status_code=status.HTTP_202_ACCEPTED
        )
    except Exception as e:
        model.model_data["status"] = "error"
        model.save()
        return {"Error": str(e)}


@router.get("/api/TrainStatus")
async def get_training_status(model_name: str):
    model = Model(model_name, minio_client, db_client)
    return {model.model_data["status"]}

@router.get("/api/GetModel")
async def get_model(model_name: str):
    model = Model(model_name, minio_client, db_client)
    return model.model_data

@router.get("/")
async def home(request: Request):
    return {"msg": "models-service"}
