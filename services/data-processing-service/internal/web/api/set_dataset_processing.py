from fastapi import HTTPException, Request
import os
import uuid
from pkg.minio.client import MinioClient
from pkg.storage.datasets.model import DatasetsStorageClient
from internal.web.api.constants import *
from internal.data.model import *
import pandas as pd

async def SetDatasetProcessing(filename: str, request: Request, strats: Dict[str, Any],minio_client: MinioClient, db_client: DatasetsStorageClient):
    md = db_client.get_object_by_name(filename)
    dataset = Dataset(minio_client, filename, md["data"]['target'])
    await dataset.initialize()
    dataset.set_data(md)
    dataset.set_strats(strats)
    db_client.update_object(filename, dataset.get_data())
    return {"message":"ok"}

async def SetDatasetAugments(filename: str, request: Request, augments: Dict[str, Any],minio_client: MinioClient, db_client: DatasetsStorageClient):
    md = db_client.get_object_by_name(filename)
    dataset = Dataset(minio_client, filename, md["data"]['target'])
    await dataset.initialize()
    dataset.set_data(md)
    dataset.set_augments(augments)
    db_client.update_object(filename, dataset.get_data())
    return {"message":"ok"}
    
async def SetDatasetSamples(filename: str, request: Request, n_samples:int,minio_client: MinioClient, db_client: DatasetsStorageClient):
    md = db_client.get_object_by_name(filename)
    dataset = Dataset(minio_client, filename, md["data"]['target'])
    await dataset.initialize()
    dataset.set_data(md)
    dataset.set_aug_num(n_samples)
    db_client.update_object(filename, dataset.get_data())
    return {"message":"ok"}
    