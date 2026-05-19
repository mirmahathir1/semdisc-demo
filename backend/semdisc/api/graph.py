from __future__ import annotations

from flask import Blueprint, jsonify, request

from ..services import graph_payload


graph_bp = Blueprint("graph", __name__, url_prefix="/api")


@graph_bp.get("/datasets/<dataset_id>/graph")
def dataset_graph(dataset_id: str):
    min_edge_score = request.args.get("minEdgeScore")
    return jsonify(
        graph_payload(
            dataset_id,
            min_edge_score=float(min_edge_score) if min_edge_score is not None else None,
        )
    )

