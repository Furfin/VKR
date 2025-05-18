from fastapi import HTTPException, Request
import os
import uuid
from pkg.minio.client import MinioClient
from pkg.storage.datasets.model import DatasetsStorageClient
from internal.web.api.constants import *
from internal.data.model import *
import pandas as pd

def GetDataset(filename: str, request: Request, db_client: DatasetsStorageClient):    
    return db_client.get_object_by_name(filename)

def GetDatasets(db_client: DatasetsStorageClient):    
    return db_client.list_objects()