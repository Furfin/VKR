from fastapi import FastAPI
from prometheus_client import make_asgi_app, Counter

app = FastAPI()
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

REQUEST_COUNT = Counter(
    'app_request_count',
    'Application Request Count',
    ['method', 'endpoint']
)

@app.get("/")
async def root():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    return {"message": "Hello World i preprocess data"}