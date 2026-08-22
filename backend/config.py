"""API-layer configuration. Domain settings remain in root config.py."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

API_HOST = os.getenv("NEURAX_API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("NEURAX_API_PORT", "8000"))

# Local frontend only — never use wildcard in production defaults
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "NEURAX_CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
]

UPLOAD_DIR = Path(os.getenv("NEURAX_UPLOAD_DIR", str(PROJECT_ROOT / "data" / "uploads")))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_MB = int(os.getenv("NEURAX_MAX_UPLOAD_MB", "100"))
