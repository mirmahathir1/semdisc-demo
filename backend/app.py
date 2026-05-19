from __future__ import annotations

from flask import Flask, jsonify
from werkzeug.exceptions import HTTPException

from config import Config
from semdisc.api import register_blueprints
from semdisc.services import ApiError


def create_app(config_object: type[Config] = Config) -> Flask:
    app = Flask(__name__)
    app.config.from_object(config_object)

    try:
        from flask_cors import CORS

        CORS(app, resources={r"/api/*": {"origins": "*"}})
    except ImportError:
        pass

    register_blueprints(app)
    register_error_handlers(app)
    return app


def register_error_handlers(app: Flask) -> None:
    @app.errorhandler(ApiError)
    def handle_api_error(error: ApiError):
        return jsonify({"error": {"code": error.code, "message": error.message}}), error.status

    @app.errorhandler(HTTPException)
    def handle_http_error(error: HTTPException):
        return (
            jsonify({"error": {"code": error.name.lower().replace(" ", "_"), "message": error.description}}),
            error.code or 500,
        )

    @app.errorhandler(Exception)
    def handle_unexpected_error(error: Exception):
        app.logger.exception("Unhandled API error")
        return jsonify({"error": {"code": "internal_error", "message": "An unexpected error occurred."}}), 500


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

