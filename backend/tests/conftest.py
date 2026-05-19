from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Use the deterministic raw-MinHash backend in tests so the suite stays fast
# and does not download the SBERT model. Must be set before semdisc imports.
os.environ.setdefault("SEMDISC_SEMANTIC_BACKEND", "raw")

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


@pytest.fixture()
def app():
    from app import create_app

    app = create_app()
    app.config.update(TESTING=True)
    return app


@pytest.fixture()
def client(app):
    return app.test_client()

