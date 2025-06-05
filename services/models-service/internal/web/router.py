from fastapi import APIRouter, Request, status, BackgroundTasks
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, Response
from pkg.minio.client import MinioClient
import pkg.storage.models.model as storage
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from internal.models.model import *
from internal.data.model import *
from fastapi.templating import Jinja2Templates
from typing import Dict, Any, List
import os
import uuid
from typing import Optional
import asyncio
from threading import Thread
import threading

minio_client = MinioClient(os.getenv("MINIO_ENDPOINT"),os.getenv("MINIO_ACCESS_KEY"),os.getenv("MINIO_SECRET_KEY"),False)

db_client = storage.ModelsStorageClient(os.getenv('DATABASE_URL'))

target_processing_tasks: Dict[str, asyncio.Task] = {}
model_statuses: Dict[str, str] = {}


router = APIRouter()

@router.post("/api/CreateModel")
async def create_model(model_name: str, dataset: str):
    try:
        resp = requests.post(API_URL+"GetDataset", params={"filename":dataset})
        if resp.json() == None:
            raise HTTPException(status_code=404, detail="dataset not found")
        Model(model_name, minio_client, db_client, dataset)
        return {"msg": "ok"}
    except Exception as e:
        return Response(status_code=400, content=json.dumps({"Error":"invalid dataset name"}))

@router.get("/api/ListModel")
async def list_models():
    try:
        return db_client.list_objects()
    except Exception as e:
        return {"Error": str(e)}

@router.post("/api/TrainModel")
def train_model( background_tasks: BackgroundTasks, model_name: str, num_epoch:int = 100):
    model = None
    try:
        model = Model(model_name, minio_client, db_client)
        
        #background_tasks.add_task(model.train_model_with,filename=model.model_data["dataset_name"],num_epoch=num_epoch)
        t = Thread(target=asyncio.run, args=(model.train_model_with(model.model_data["dataset_name"], num_epoch),))
        t.daemon = True
        t.start()

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
        print(e)
        return {"Error": str(e)}


@router.get("/api/TrainStatus")
async def get_training_status(model_name: str):
    model = Model(model_name, minio_client, db_client)
    return {model.model_data["status"]}

@router.get("/api/GetModel")
async def get_model(model_name: str):
    model = Model(model_name, minio_client, db_client)
    return model.model_data

@router.post("/api/Delete")
async def delete_model(model_name: str):
    try:
        db_client.delete_object(model_name)
        
        return {"message":"ok"}
    except Exception as e:
        return {"Error": str(e)}
    
def load_model_from_url(model_url):
    # Create a temporary file that won't be automatically deleted
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, 'model.keras')
    
    try:
        # Download the model
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        
        # Write to temporary file
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Load model
        model = keras.models.load_model(temp_path)
        return model
    finally:
        # Clean up - remove the temporary file
        try:
            os.remove(temp_path)
            os.rmdir(temp_dir)
        except:
            pass

async def predict_data(modelData: Model, model_name: str, filename: str):
    try:
        data = Data(minio_client, filename)
        await data.initialize(modelData.model_data["dataset_md"])
        data_url = await data.upload_preprocessed_dataset()
        X = pd.read_csv(await modelData.minio_client.download_fileobj("data", "preprocessed_"+filename))
        url = await minio_client.get_url("models", model_name+".keras")
        model = load_model_from_url(url)
        print(X.values, X.shape)
        predictions = model.predict(X.values)
        X['prediction'] = predictions
        
        buffer = BytesIO()
        X.to_csv(buffer, index=False)
        buffer.seek(0)
        content_type = "text/csv"
        await minio_client.upload_fileobj(
            bucket_name=DATA_BUCKET,
            object_name="model_predicted_"+filename,
            file=buffer,
            length=buffer.getbuffer().nbytes,
            content_type=content_type
        )
        modelData.model_data["data_status"] = "done"
        modelData.model_data["data_file"] = "model_predicted_"+filename
        modelData.save()
    except Exception as e:
        print(e)
        modelData.model_data["data_status"] = "error"
        modelData.save()

@router.post("/api/ProcessNewData")
def new_data_process(background_tasks: BackgroundTasks, model_name: str, filename: str):
    try:
        model = Model(model_name, minio_client, db_client)
        if model.model_data["status"] != "done":
            raise HTTPException
        model.model_data["data_status"] = "running"
        model.save()
        background_tasks.add_task(predict_data,modelData = model, model_name=model_name,filename=filename)
        return Response(status_code=202)
        
    except Exception as e:
        return Response(status_code=400)
@router.get("/")
async def home(request: Request):
    return {"msg": "models-service"}

@router.get('/metrics')
def metrics():
    """ Exposes only explicitly registered metrics. """
    return Response(generate_latest(REGISTRY), status_code=200,headers= {'Content-Type': CONTENT_TYPE_LATEST})

