from fastapi import APIRouter, Request
from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from pkg.minio.client import MinioClient
from internal.web.api.create_dataset import CreateDataset
from internal.web.api.get_dataset import GetDataset, GetDatasets
from internal.web.api.set_dataset_processing import SetDatasetProcessing, SetDatasetAugments, SetDatasetSamples
from pkg.storage.datasets.model import DatasetsStorageClient
from internal.data.preprocess_dataset import *
from internal.data.dataset_augmentation import *
from internal.web.api.constants import *
from internal.data.model import *
from typing import Dict, Any, List
import asyncio
import os
import uuid
from concurrent.futures import ProcessPoolExecutor

minio_client = MinioClient(os.getenv("MINIO_ENDPOINT"),os.getenv("MINIO_ACCESS_KEY"),os.getenv("MINIO_SECRET_KEY"),False)

db_client = DatasetsStorageClient(os.getenv('DATABASE_URL'))

processing_tasks: Dict[str, asyncio.Task] = {}
target_processing_tasks: Dict[str, asyncio.Task] = {}
file_statuses: Dict[str, str] = {}  # "dataset_name": "processing|ready|error"

router = APIRouter()

@router.post("/api/CreateDataset")
async def create_dataset(filename: str, target_column: str):
    try:
        result = await CreateDataset(filename,target_column, minio_client, db_client)
        return result
    except Exception as e:
        return {"Error": str(e)}

@router.post("/api/GetDataset")
async def get_dataset(request: Request, filename: str):
    try:
        result = GetDataset(filename, request, db_client)
        return result
    except Exception as e:
        return {"Error": str(e)}

@router.get("/api/GetPreprocessingOptions")
async def GetPreprocessingOptions(request: Request):
    return {"data":list(strats.keys)}

@router.get("/api/GetAugmentationOptions")
async def GetAugmentationOptions(request: Request):
    return {"data":list(augmentations.keys())}

@router.post("/api/SetDatasetPreprocessing")
async def SetDatasetPreprocessing(request: Request, filename: str, strats: Dict[str, Any]):
    try:
        result = await SetDatasetProcessing(filename, request, strats, minio_client, db_client)
        return result
    except Exception as e:
        return {"Error": str(e), "strats":strats}

@router.post("/api/SetDatasetAugmentation")
async def SetDatasetAugmentation(request: Request, filename: str, augments: Dict[str, Any]):
    try:
        result = await SetDatasetAugments(filename, request, augments, minio_client, db_client)
        return result
    except Exception as e:
        return {"Error": str(e)}

@router.post("/api/AddAugmentation")
async def SetDatasetAugmentationNum(request: Request, filename: str, n_samples: int):
    try:
        result = await SetDatasetSamples(filename, request, n_samples, minio_client, db_client)
        return result
    except Exception as e:
        return {"Error": str(e)}

@router.post("/api/GetPreprocessedDataset")
async def get_preproccessed_dataset(filename: str):
    try:
        md = db_client.get_object_by_name(filename)
        dataset = Dataset(minio_client, filename, md["data"]['target'])
        await dataset.initialize()
        dataset.set_data(md)
        data = await dataset.get_preprocessed_dataset()
        
        return StreamingResponse(
            iter([data.to_csv(index=False)]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=data.csv"}
        )
    except Exception as e:
        return {"Error": str(e)}

@router.post("/api/GetPreprocessedTarget")
async def get_preprocessed_target(filename: str):
    try:
        md = db_client.get_object_by_name(filename)
        dataset = Dataset(minio_client, filename, md["data"]['target'])
        await dataset.initialize()
        dataset.set_data(md)
        data = await dataset.get_preprocessed_target()
        
        return StreamingResponse(
            iter([data.to_csv(index=False)]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=data.csv"}
        )
    except Exception as e:
        return {"Error": str(e)}

@router.get("/api/ListDatasets")
async def list_datasets():
    try:
        result = GetDatasets(db_client)
        return result
    except Exception as e:
        return {"Error": str(e)}
    
@router.post("/api/DropColumn")
async def drop_column(filename: str, column: str):
    try:
        md = db_client.get_object_by_name(filename)
        dataset = Dataset(minio_client, filename, md["data"]['target'])
        await dataset.initialize()
        dataset.set_data(md)
        dataset.drop_column(column)
        db_client.update_object(filename, dataset.get_data())
        
        return {"message":"ok"}
    except Exception as e:
        return {"Error": str(e)}

@router.post("/api/CastToCat")
async def cast_to_cat(filename: str, column: str):
    try:
        md = db_client.get_object_by_name(filename)
        dataset = Dataset(minio_client, filename, md["data"]['target'])
        await dataset.initialize()
        dataset.set_data(md)
        dataset.cast_to_cat(column)
        db_client.update_object(filename, dataset.get_data())
        
        return {"message":"ok"}
    except Exception as e:
        return {"Error": str(e)}

@router.post("/api/Delete")
async def cast_to_cat(filename: str):
    try:
        db_client.delete_object(filename)
        
        return {"message":"ok"}
    except Exception as e:
        return {"Error": str(e)}

async def process_dataset(
    filename: str,
    dataset: Dataset,
):
    try:
        dataset.data["status"] = "processing"
        db_client.update_object(filename, dataset.get_data())
        await dataset.upload_preprocessed_dataset()
        await dataset.upload_preprocessed_target()
        dataset.data["status"] = "done"
        db_client.update_object(filename, dataset.get_data())
    except Exception as e:
        dataset.data["status"] = "done"
        db_client.update_object(filename, dataset.get_data())
        file_statuses[filename] = f"error: {str(e)}"
        raise
    finally:
        # Очистка задачи
        if filename in processing_tasks:
            file_statuses[filename] = "done"
            del processing_tasks[filename]

@router.post("/start_processing_dataset")
async def start_processing(filename: str, background_tasks: BackgroundTasks):
    md = db_client.get_object_by_name(filename)
    dataset = Dataset(minio_client, filename, md["data"]['target'])
    await dataset.initialize()
    dataset.set_data(md)
    
    if filename in processing_tasks:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Dataset is already being processed"
        )
    
    processing_task = "task"
    
    background_tasks.add_task(process_dataset,filename, dataset)

    processing_tasks[filename] = processing_task
    file_statuses[filename] = "processing"
    
    return JSONResponse(
        content={
            "message": "Processing started",
            "dataset_name": filename,
            "aug_num": dataset.get_data()["aug_num"],
        },
        status_code=status.HTTP_202_ACCEPTED
    )

@router.get("/processing_status")
async def get_processing_status(filename: str):
    if filename not in file_statuses:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    status_info = {
        "dataset_name": filename,
        "status": file_statuses[filename]
    }
    
    return status_info