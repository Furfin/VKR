import pandas as pd
from pkg.minio.client import MinioClient
import pkg.storage.models.model as storage
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import autokeras as ak
import keras
from io import BytesIO
import json
import tempfile
import os


API_URL = "http://localhost:8001/api/"
MODELS_BUCKET = "models"

class Model():
    
    def __init__(self,model_name: str, minio_client: MinioClient, db_client: storage.ModelsStorageClient):
        self.minio_client = minio_client
        self.db = db_client
        self.model_name = model_name
        try:
            self.model_data = {"name":model_name}
            self.model_data["status"] = "new"
            self.db.create_object(model_name, self.model_data)
        except Exception as e:
            try:
                self.model_data = db_client.get_object_by_name(model_name)["data"]
            except:
                raise
    
    async def train_model_with(self, filename:str, num_epoch: int):
        resp = requests.post(API_URL+"GetDataset", params={"filename":filename})
        data = dict(resp.json())["data"]
        self.model_data["dataset_md"] = data
        self.model_data["status"] = "training"
        self.save()
        
        if data["columns"][data["target"]]["type"] == "numeric":
            model, result = await self.numeric_train(filename, num_epoch)
        elif data["columns"][data["target"]]["type"] == "cat":
            model, result = await self.cat_train(filename, num_epoch)
        else:
            raise ValueError("invalid target type:"+ data["columns"][data["target"]]["type"])
        filepath = "tmp/"+self.model_name+'.keras'
        model.save(filepath)
        with open(filepath, "rb") as file_data:
            file_stat = os.stat(filepath)
            await self.minio_client.upload_fileobj(
                    bucket_name=MODELS_BUCKET,
                    object_name=self.model_name+".keras",
                    file=file_data,
                    length=file_stat.st_size,
                    content_type="keras/model"
                )
        if os.path.exists(filepath):
            os.remove(filepath)
        self.model_data["last_result"] = result
        self.model_data["status"] = "done"
        self.save()
    
    def save(self):
        self.db.update_object(self.model_name, self.model_data)
    
    async def numeric_train(self, filename:str, num_epoch: int):
        X = pd.read_csv(await self.minio_client.download_fileobj("datasets", "preprocessed_"+filename))
        Y = pd.read_csv(await self.minio_client.download_fileobj("datasets", "preprocessed_target_"+filename))
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=1)
        input  = ak.Input()
        out = ak.RegressionHead(output_dim=1)
        search = ak.AutoModel(max_trials=10, overwrite=True, inputs=input, outputs=out,)
        search.fit(x = X_train.values, y = y_train.values, epochs=num_epoch, verbose = 0)
        model = search.export_model()
        return model, model.evaluate(X_test, y_test)
    
    async def cat_train(self, filename:str, num_epoch: int):
        X = pd.read_csv(await self.minio_client.download_fileobj("datasets", "preprocessed_"+filename))
        Y = pd.read_csv(await self.minio_client.download_fileobj("datasets", "preprocessed_target_"+filename))

        classes = Y.nunique()[0]
        encoder = LabelEncoder()
        Y = pd.Series(
                encoder.fit_transform(Y),
                index=Y.index,
            )
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=1)

        input  = ak.Input()
        out = ak.ClassificationHead(num_classes=classes,  loss=keras.losses.CategoricalCrossentropy())
        search = ak.AutoModel(max_trials=1, overwrite=True, inputs=input, outputs=out)
        search.fit(x=X_train.values, y=y_train.values, epochs=num_epoch, verbose = 0)
        model = search.export_model()
        return model, list(model.evaluate(X_test, y_test))
    
    async def save_model_to_minio(self, model):
        try:
            with tempfile.NamedTemporaryFile(suffix=".keras", delete=False, mode="wb+") as tmp_file:
                temp_path = tmp_file.name
                model.save(temp_path)
                tmp_file.seek(0)
                tmp_file.flush()
                file_size = tmp_file.tell()
                print(file_size)
                tmp_file.seek(0)
                await self.minio_client.upload_fileobj(
                    bucket_name=MODELS_BUCKET,
                    object_name=self.model_name+".keras",
                    file=tmp_file,
                    length=file_size,
                    content_type="keras/model"
                )
        except Exception as e:
            raise ValueError("error error")
        finally:
            # 5. Гарантированное удаление временного файла
            if os.path.exists(temp_path):
                os.remove(temp_path)
        