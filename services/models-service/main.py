# app/main.py
from fastapi import FastAPI
import internal.web.router as r
from fastapi.staticfiles import StaticFiles
from prometheus_client import make_asgi_app, Counter, generate_latest

app = FastAPI()

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

app.include_router(r.router)