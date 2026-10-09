from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router
from backend.model.mlp import VocalToneModel


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Prepare the MLP before the first request reaches the API."""

    vocaltone_model = VocalToneModel(storage_dir="backend/model/artifacts")
    vocaltone_model.load_or_train()
    app.state.vocaltone_model = vocaltone_model
    yield


app = FastAPI(
    title="VocalTone AI Backend",
    description="Analiza voz en archivos WAV y recomienda una tonalidad para cantar.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok", "service": "VocalTone AI"}
