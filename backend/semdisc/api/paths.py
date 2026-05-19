from __future__ import annotations

from flask import Blueprint, jsonify, request

from ..services import materialize_payload, navigation_paths_payload


paths_bp = Blueprint("paths", __name__, url_prefix="/api")


@paths_bp.get("/datasets/<dataset_id>/paths")
def dataset_paths(dataset_id: str):
    return jsonify(
        navigation_paths_payload(
            dataset_id,
            max_hops=int(request.args.get("maxHops", 4)),
            k=int(request.args.get("k", 10)),
        )
    )


@paths_bp.post("/materialize")
def materialize():
    return jsonify(materialize_payload(request.get_json(silent=True) or {}))

