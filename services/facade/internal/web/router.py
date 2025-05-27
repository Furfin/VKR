from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pkg.minio.client import MinioClient
from pkg.storage.datasets.model import *
from pkg.storage.models.model import *
from fastapi import FastAPI, UploadFile, File, Request, HTTPException, Form, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import uuid
from typing import Optional
from . import upload
import requests
import autokeras as ak
import keras
from io import BytesIO
import tempfile

minio_client = MinioClient(os.getenv("MINIO_ENDPOINT"),os.getenv("MINIO_ACCESS_KEY"),os.getenv("MINIO_SECRET_KEY"),False)
db_client = DatasetsStorageClient(os.getenv('DATABASE_URL'))

models_db_client = ModelsStorageClient(os.getenv('DATABASE_URL'))

router = APIRouter()
templates = Jinja2Templates(directory="internal/templates")

DATASET_API_URL = "http://localhost:8001/api/"

MODELS_API_URL = "http://localhost:8002/api/"

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
async def datasets(request: Request, newfile: str = ""):
    datasets = db_client.list_objects()
    return templates.TemplateResponse("list.html", {"request": request, "objects": datasets, "newfile": newfile})

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

@router.post("/update-value")
async def update_aug(filename: str, number: int = Form(...), ):
    aug = number
    if aug < 0:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Invalid value for augmentation"
        )
    requests.post(DATASET_API_URL+"AddAugmentation", params={"filename":filename,"n_samples":aug})
    return RedirectResponse(f"http://localhost:8000/datasets/{filename}", status_code=303)

@router.post("/update_strat/{filename}/{column}")
async def update_aug(filename: str, column:str,strat: str ):
    requests.post(DATASET_API_URL+"UpdateStart", params={"filename":filename,"column":column,"strat":strat})
    return RedirectResponse(f"http://localhost:8000/datasets/{filename}", status_code=303)

@router.post("/update_aug/{filename}/{column}")
async def update_aug(filename: str, column:str,strat: str ):
    requests.post(DATASET_API_URL+"UpdateAug", params={"filename":filename,"column":column,"strat":strat})
    return RedirectResponse(f"http://localhost:8000/datasets/{filename}", status_code=303)

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










#################################################################################################################################



@router.get("/models", response_class=HTMLResponse)
async def models(request: Request, newmodel: str = "", error: str = ""):
    models = models_db_client.list_objects()
    return templates.TemplateResponse("models_list.html", {"request": request, "objects": models, "newmodel": newmodel,"error":error})


@router.post("/create_model")
async def create_model(model_name: str = Form(...),dataset: str = Form(...), ):
    
    resp = requests.post(MODELS_API_URL+"CreateModel", params={"model_name":model_name,"dataset":dataset})
    if resp.status_code != 200:
        return RedirectResponse(f"http://localhost:8000/models?error=CreateModelError", status_code=303)
    return RedirectResponse(f"http://localhost:8000/models?newmodel={model_name}", status_code=303)


def load_model_from_url(model_url):
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, 'model.keras')
    
    try:
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        model = keras.models.load_model(temp_path)
        return model
    finally:
        try:
            os.remove(temp_path)
            os.rmdir(temp_dir)
        except:
            pass

@router.get("/models/{model_name}", response_class=HTMLResponse)
async def model_detail(request: Request, model_name: str, error:str = ""):
    obj = models_db_client.get_object_by_name(model_name)
    data = obj["data"]
    model_info = {}
    url = ""
    if data["status"] == "done":
        url = await minio_client.get_url("models", model_name+".keras")
        model = load_model_from_url(url)
        print(model.summary())
        model_info = {
            "name": model.name,
            "input_shape": model.input_shape,
            "output_shape": model.output_shape,
            "trainable_params": model.count_params(),
            "layers": [],
            "config": model.get_config()
        }
        # Extract layer information
        for layer in model.layers:
            layer_info = {
                "name": layer.name,
                "type": layer.__class__.__name__,
                "input_shape": layer.input_spec,
                "trainable": layer.trainable,
                "params": layer.count_params(),
                "config": layer.get_config()
            }
            model_info["layers"].append(layer_info)
    pred_url = ""
    if data["data_status"] == "done":
        pred_url = await minio_client.get_url("data",data["data_file"])
            
    if not obj:
        return RedirectResponse("/")
    return templates.TemplateResponse("model_detail.html", {"request": request,"model_name":model_name,"dataset":data["dataset_name"],"status":data["status"],
                                                      "model_url":url, "error": error, "model":model_info, "model_run_status": data["data_status"], "pred_url": pred_url
                                                      })
    


@router.post("/generate_model/{model_name}")
async def create_model(model_name: str,enum: int = Form(...) ):
    if enum <= 0:
        return RedirectResponse(f"http://localhost:8000/models/"+model_name+"?error=invalidEpochsNum",  status_code=303)
    resp = requests.post(MODELS_API_URL+"TrainModel", params={"model_name":model_name,"num_epoch":enum})
    if resp.status_code != 202:
        return RedirectResponse(f"http://localhost:8000/models/"+model_name+"?error=TrainModelError", status_code=303)
    return RedirectResponse(f"http://localhost:8000/models/"+model_name, status_code=303)


@router.post("/run_model/{model_name}")
async def handle_model_nem_data(
    model_name: str,
    file: UploadFile = File(...),
):
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)

    if file_size > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File too large (max 10MB)"
        )

    file_ext = os.path.splitext(file.filename)[1]
    if file_ext != ".csv":
        return RedirectResponse(f"http://localhost:8000/models/"+model_name+"?error=IvalidDataUploaded", status_code=303)
    unique_filename = f"{uuid.uuid4().hex}{file_ext}"
    await minio_client.upload_fileobj(
        bucket_name="data",
        object_name=unique_filename,
        file=file.file,
        length=file_size,
        content_type=file.content_type
    )
    requests.post(MODELS_API_URL+"ProcessNewData", params={"model_name":model_name, "filename":unique_filename})
    return RedirectResponse(f"http://localhost:8000/models/"+model_name, status_code=303)
