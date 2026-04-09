import os
from fastapi.middleware.cors import CORSMiddleware
from backend_main import app

raw = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = [item.strip() for item in raw.split(",") if item.strip()] or ["*"]
allow_all = allowed_origins == ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=not allow_all,
    allow_methods=["*"],
    allow_headers=["*"],
)
