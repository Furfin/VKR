import pandas as pd
from typing import Dict, Any, List
from pkg.minio.client import MinioClient
from internal.data.dataset_import import *
from internal.data.preprocess_dataset import *
from io import BytesIO

DATA_BUCKET = "data"

class Data:    
    def __init__(self,minio_client: MinioClient, filename: str):
        self.filename = filename
        self.minio_client = minio_client
        self.df = None
        self.data: Dict[str, Any] = {"aug_num":0, "status":"new"}
        
    async def initialize(self, data):
        self.df = clean_dataframe(await self.minio_client.download_fileobj_sample(DATA_BUCKET, self.filename))
        self.data, self.df = detect_column_types(self.df,self.data, 0.3)
        self.set_data(data)
    
    def set_data(self, data: Dict[str, Any]):
        self.data = data
        return self.data
        
    def get_data(self) -> Dict[str, Any]:
        return self.data
    
    async def upload_preprocessed_dataset(self):
        df = clean_dataframe(pd.read_csv(await self.minio_client.download_fileobj(DATA_BUCKET, self.filename)))
        try:
            df.drop(self.data["target"], axis=1, inplace=True)
            del self.data[COLUMNS_KEY][self.data["target"]]
        except:
            pass
        columns = self.data[COLUMNS_KEY].keys()
        data = self.data[COLUMNS_KEY]
        for col in columns:
            if col == self.data["target"]:
                continue
            df = apply_strat(df, col, strats[data[col]["strat"]])
        
        buffer = BytesIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        content_type = "text/csv"
        await self.minio_client.upload_fileobj(
            bucket_name=DATA_BUCKET,
            object_name="preprocessed_"+self.filename,
            file=buffer,
            length=buffer.getbuffer().nbytes,
            content_type=content_type
        )
        return await self.minio_client.get_url(DATA_BUCKET, "preprocessed_"+self.filename)