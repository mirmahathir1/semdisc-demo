from __future__ import annotations

from flask import Blueprint, jsonify, request

from ..data_loader import load_datalake_from_uploads
from ..serialization import table_summaries
from ..services import (
    ApiError,
    bundled_datalake_payload,
    bundled_datalakes_payload,
    create_dataset,
    demo_datalake_payload,
    preprocess_dataset,
    semantic_types_payload,
    table_preview,
    _dataset_or_404,
)


datasets_bp = Blueprint("datasets", __name__, url_prefix="/api")


@datasets_bp.get("/demo/datalake")
def demo_datalake():
    return jsonify(demo_datalake_payload())


@datasets_bp.get("/datalakes")
def bundled_datalakes():
    return jsonify(bundled_datalakes_payload())


@datasets_bp.get("/datalakes/<name>")
def bundled_datalake(name: str):
    return jsonify(bundled_datalake_payload(name))


@datasets_bp.post("/datasets")
def create_dataset_endpoint():
    uploads = []
    for file in request.files.getlist("files") or request.files.values():
        uploads.append((file.filename, file.read()))
    if not uploads:
        raise ApiError("bad_request", "Upload one or more CSV files, or a ZIP containing CSV files.", 400)
    tables = load_datalake_from_uploads(uploads)
    state = create_dataset(tables, source="uploaded")
    return jsonify({"datasetId": state.id, "tables": table_summaries(state.tables)}), 201


@datasets_bp.post("/preprocess")
def preprocess_endpoint():
    payload = request.get_json(silent=True) or {}
    dataset_id = payload.get("datasetId")
    if not dataset_id:
        raise ApiError("bad_request", "datasetId is required.", 400)
    state = preprocess_dataset(str(dataset_id), payload.get("hyperparameters") or payload)
    return jsonify(
        {
            "datasetId": state.id,
            "status": "ready",
            "preprocessed": True,
            "hyperparameters": state.hyperparameters,
            "tables": table_summaries(state.tables),
            "edgeCount": sum(len(edges) for edges in state.graph.values()) // 2,
        }
    )


@datasets_bp.get("/datasets/<dataset_id>/tables")
def dataset_tables(dataset_id: str):
    state = _dataset_or_404(dataset_id)
    return jsonify({"datasetId": dataset_id, "tables": table_summaries(state.tables)})


@datasets_bp.get("/datasets/<dataset_id>/tables/<table_name>")
def dataset_table(dataset_id: str, table_name: str):
    return jsonify(
        table_preview(
            dataset_id,
            table_name,
            page=int(request.args.get("page", 1)),
            page_size=int(request.args.get("pageSize", 100)),
        )
    )


@datasets_bp.get("/datasets/<dataset_id>/semantic-types")
def semantic_types(dataset_id: str):
    return jsonify(semantic_types_payload(dataset_id))
