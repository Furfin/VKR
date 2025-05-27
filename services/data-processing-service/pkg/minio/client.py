from minio import Minio
from minio.error import S3Error
import os
from dotenv import load_dotenv
from fastapi import HTTPException
import aiohttp
from io import BytesIO, StringIO
from datetime import timedelta
import pandas as pd

load_dotenv()

class MinioClient:
    def __init__(self, endpoint: str, access_key: str, secret_key: str, secure:bool):
        self.client = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )
    
    async def upload_fileobj(self, bucket_name: str, object_name: str, file, length: int, content_type: str):
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
            
            self.client.put_object(
                bucket_name,
                object_name,
                file,
                length,
                content_type=content_type
            )
            return object_name
        except S3Error as e:
            raise HTTPException(
                status_code=500,
                detail=f"MinIO upload error: {str(e)}"
            )
    
    async def download_fileobj(self, bucket_name: str, object_name: str) -> BytesIO:
        try:
            url = self.client.presigned_get_object(bucket_name, object_name, expires=timedelta(seconds=36000))
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        raise HTTPException(status_code=response.status, detail="Failed to download file")
                    data = await response.read()
                    return BytesIO(data)
        except S3Error as e:
            raise HTTPException(status_code=500, detail=f"MinIO download error: {str(e)}")
        
        
    async def download_fileobj_sample(self, bucket_name: str, object_name: str) -> pd.DataFrame:
        try:
            url = self.client.presigned_get_object(
                bucket_name, 
                object_name, 
                expires=timedelta(seconds=36000)
            )
            
            lines = []
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        raise HTTPException(
                            status_code=response.status,
                            detail="Failed to download file"
                        )
                    buffer = ""
                    async for chunk in response.content.iter_any():
                        buffer += chunk.decode('utf-8')
                        *complete_lines, partial_line = buffer.split('\n')
                        buffer = partial_line
                        
                        lines.extend(complete_lines)
                        if len(lines) >= 100:
                            break
            csv_content = '\n'.join(lines[:100])
            return pd.read_csv(StringIO(csv_content))
        
        except S3Error as e:
            raise HTTPException(
                status_code=500,
                detail=f"MinIO download error: {str(e)}"
            )
        except pd.errors.EmptyDataError:
            raise HTTPException(
                status_code=400,
                detail="CSV file appears to be empty"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error: {str(e)}"
            )
            
        
    async def get_url(self, bucket_name: str, object_name: str) -> str:
        try:
            file_url = self.client.presigned_get_object(
                bucket_name,
                object_name,
                expires=timedelta(hours=2)
            )
            
            return file_url
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"MinIO error: {str(e)}")