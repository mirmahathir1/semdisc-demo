from __future__ import annotations

from flask import Blueprint, jsonify, request

from ..services import ApiError, demo_query_payload, query_paths_payload, submit_query


query_bp = Blueprint("query", __name__, url_prefix="/api")


@query_bp.get("/demo/query")
def demo_query():
    return jsonify(demo_query_payload(request.args.get("datalake")))


@query_bp.post("/query")
def query_submit():
    payload = request.get_json(silent=True) or {}
    dataset_id = payload.get("datasetId")
    if not dataset_id:
        raise ApiError("bad_request", "datasetId is required.", 400)
    qstate = submit_query(str(dataset_id), payload)
    return jsonify(
        {
            "queryId": qstate.id,
            "datasetId": qstate.dataset_id,
            "rows": int(qstate.query.shape[0]),
            "columns": [str(c) for c in qstate.query.columns],
        }
    ), 201


@query_bp.get("/query/<query_id>/paths")
def query_paths(query_id: str):
    return jsonify(
        query_paths_payload(
            query_id,
            max_hops=int(request.args.get("maxHops", 4)),
            k=int(request.args.get("k", 10)),
        )
    )
