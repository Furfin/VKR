# app/main.py
from fastapi import FastAPI
import internal.web.router as r
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.mount("/static", StaticFiles(directory="internal/templates/static"), name="static")

app.include_router(r.router)