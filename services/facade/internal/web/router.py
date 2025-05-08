from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from internal.minio.client import MinioClient
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import uuid
from typing import Optional
from . import upload

minio_client = MinioClient()

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

@router.get("/datasets", response_class=HTMLResponse)
async def datasets(request: Request):
    return templates.TemplateResponse(
        "datasets.html",
        {"request": request}
    )
    
@router.post("/upload")
async def handle_upload(
    request: Request,
    file: UploadFile = File(...),
    bucket_name: Optional[str] = None
):
    try:
        result = await upload.Upload(file, request, bucket_name, minio_client, templates)
        return result
    except Exception as e:
        return {"Error": str(e)}
