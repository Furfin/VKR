from fastapi import HTTPException, Request
import os
import uuid
from pkg.minio.client import MinioClient
from pkg.storage.datasets.model import DatasetsStorageClient
from internal.web.api.constants import *
from internal.data.model import *
import pandas as pd

async def CreateDataset(filename: str, traget: str, minio_client: MinioClient, db_client: DatasetsStorageClient):
    dataset = Dataset(minio_client, filename, traget)
    await dataset.initialize()
    db_client.create_object(filename, dataset.get_data())
        
    return {"message":"ok"}