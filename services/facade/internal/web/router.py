from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pkg.minio.client import MinioClient
from pkg.storage.datasets.model import *
from fastapi import FastAPI, UploadFile, File, Request, HTTPException, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import uuid
from typing import Optional
from . import upload

minio_client = MinioClient(os.getenv("MINIO_ENDPOINT"),os.getenv("MINIO_ACCESS_KEY"),os.getenv("MINIO_SECRET_KEY"),False)
db_client = DatasetsStorageClient(os.getenv('DATABASE_URL'))

router = APIRouter()
templates = Jinja2Templates(directory="internal/templates")

@router.on_event("startup")
async def startup_event():
    """Initialize MinIO bucket if needed"""
    try:
        bucket = os.getenv("DEFAULT_BUCKET")
        if not minio_client.client.bucket_exists(bucket):
            minio_client.client.make_bucket(bucket)
    except Exception as e:
        print(f"Couldn't initialize MinIO bucket: {e}")

@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "home.html", 
        {"request": request, "page_title": "Home"}
    )

@router.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse(
        "home.html",
        {"request": request, "page_title": "About Us"}
    )

@router.get("/datasets/upload", response_class=HTMLResponse)
async def datasetsUpload(request: Request):
        return templates.TemplateResponse(
        "datasets.html",
        {"request": request}
    )

@router.get("/datasets", response_class=HTMLResponse)
async def datasets(request: Request):
    datasets = db_client.list_objects()
    return templates.TemplateResponse("list.html", {"request": request, "objects": datasets})

@router.get("/datasets/{filename}", response_class=HTMLResponse)
async def object_detail(request: Request, filename: str):
    obj = db_client.get_object_by_name(filename)
    data = obj["data"]
    cols = len(obj["data"]["columns"].keys())
    target_t = data["columns"][data["target"]]["type"]
    status = data["status"]
    url1 = ""
    url2 = ""
    if status == "done":
        url1 = await minio_client.get_url("datasets", "preprocessed_"+filename)
        url2 = await minio_client.get_url("datasets", "preprocessed_target_"+filename)
    if not obj:
        return RedirectResponse("/")
    return templates.TemplateResponse("detail.html", {"request": request, "obj": obj, "obj_id": filename,
                                                      "cols":cols, "filename":filename, "tt":target_t,
                                                      "objects":obj["data"]["columns"],"status":status,
                                                      "data_url":url1, "target_url":url2
                                                      })

@router.post("/upload")
async def handle_upload(
    target: str = Form(...),
    file: UploadFile = File(...),
):
    try:
        result = await upload.Upload(file, target, minio_client, templates)
        return result
    except Exception as e:
        return {"Error": str(e)}
