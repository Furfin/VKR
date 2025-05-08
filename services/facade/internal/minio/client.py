from minio import Minio
from minio.error import S3Error
import os
from dotenv import load_dotenv
from fastapi import HTTPException

load_dotenv()

class MinioClient:
    def __init__(self):
        self.client = Minio(
            endpoint=os.getenv("MINIO_ENDPOINT"),
            access_key=os.getenv("MINIO_ACCESS_KEY"),
            secret_key=os.getenv("MINIO_SECRET_KEY"),
            secure=os.getenv("MINIO_SECURE", "False").lower() == "true"
        )
    
    async def upload_fileobj(self, bucket_name: str, object_name: str, file, length: int, content_type: str):
        try:
            # Ensure bucket exists
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
            
            # Upload directly from file object
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