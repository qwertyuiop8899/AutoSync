import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from offset_engine import OffsetEngine
import plugin_jobs


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("AUTOSYNC_DATA_DIR", APP_DIR / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "plugin_jobs.db"
raw_proxy = os.getenv("AUTOSYNC_PROXY", "").strip()
PROXY = raw_proxy.split(",")[0].strip() if raw_proxy else ""

offset_engine = OffsetEngine(proxy=PROXY)


@asynccontextmanager
async def lifespan(app: FastAPI):
    plugin_jobs.start_worker(offset_engine, DB_PATH)
    yield
    await plugin_jobs.stop_worker()


app = FastAPI(
    title="AutoSync",
    description="Dedicated Audio/Video Offset Correlation & Measurement Service for ToastFlix & PriSynx",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
cors_origins = [item.strip() for item in os.getenv("CORS_ORIGINS", "*").split(",") if item.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Mount plugin jobs router (/plugin/jobs, /plugin/status, /plugin/queue, etc.)
app.include_router(plugin_jobs.router)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "AutoSync",
        "version": "1.0.0",
    }
