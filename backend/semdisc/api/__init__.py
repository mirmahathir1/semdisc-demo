from __future__ import annotations

from flask import Flask

from .datasets import datasets_bp
from .graph import graph_bp
from .health import health_bp
from .paths import paths_bp
from .query import query_bp


def register_blueprints(app: Flask) -> None:
    app.register_blueprint(health_bp)
    app.register_blueprint(datasets_bp)
    app.register_blueprint(graph_bp)
    app.register_blueprint(paths_bp)
    app.register_blueprint(query_bp)

