FROM python:3.12-alpine

WORKDIR /app

RUN apk add --no-cache build-base
COPY req.txt .
RUN pip install --no-cache-dir -r req.txt

COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]