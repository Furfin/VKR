from fastapi import HTTPException
import os
import uuid

async def Upload(file, request, bucket_name, minio_client, templates):
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > 100 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File too large (max 100MB)"
        )

    file_ext = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4().hex}{file_ext}"

    bucket = bucket_name or os.getenv("DEFAULT_BUCKET")
    await minio_client.upload_fileobj(
        bucket_name=bucket,
        object_name=unique_filename,
        file=file.file,
        length=file_size,
        content_type=file.content_type
    )

    return templates.TemplateResponse(
        "datasets.html",
        {
            "request": request,
            "filename": unique_filename,
            "size": file_size,
            "object_name": unique_filename,
            "bucket": bucket
        }
    )