import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.main import app
from backend.services.component_registry import reset_registry


@pytest.fixture
def client():
    reset_registry()
    with TestClient(app) as c:
        yield c
    reset_registry()
