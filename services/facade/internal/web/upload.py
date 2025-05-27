from fastapi import HTTPException
from fastapi.responses import RedirectResponse
import os
import uuid
import requests

API_URL = "http://localhost:8001/api/"

async def Upload(file, target, minio_client, templates):
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File too large (max 10MB)"
        )

    file_ext = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4().hex}{file_ext}"

    bucket = os.getenv("DEFAULT_BUCKET")
    await minio_client.upload_fileobj(
        bucket_name=bucket,
        object_name=unique_filename,
        file=file.file,
        length=file_size,
        content_type=file.content_type
    )
    
    requests.post(API_URL+"CreateDataset", params={"filename":unique_filename, "target_column":target})

    return RedirectResponse(f"http://localhost:8000/datasets?newfile={unique_filename}", status_code=303)