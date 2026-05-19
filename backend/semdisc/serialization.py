from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence

import pandas as pd

from .models import EdgeInfo
from .paths import path_to_dot


def clean_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def dataframe_to_json(df: pd.DataFrame, *, page: int = 1, page_size: int = 100) -> dict:
    page = max(1, int(page))
    page_size = max(1, min(int(page_size), 1000))
    start = (page - 1) * page_size
    end = start + page_size
    preview = df.iloc[start:end]
    rows = [
        {str(col): clean_value(row[col]) for col in preview.columns}
        for _, row in preview.iterrows()
    ]
    return {
        "columns": [str(c) for c in df.columns],
        "rows": rows,
        "page": page,
        "pageSize": page_size,
        "totalRows": int(df.shape[0]),
        "totalPages": max(1, math.ceil(int(df.shape[0]) / page_size)),
    }


def edge_to_json(edge: EdgeInfo) -> dict:
    return {
        "left": edge.left,
        "right": edge.right,
        "key": edge.key,
        "rightKey": edge.right_key or edge.key,
        "score": float(edge.score),
        "joinType": edge.join_type,
    }


def edge_from_json(payload: dict) -> EdgeInfo:
    return EdgeInfo(
        left=str(payload["left"]),
        right=str(payload["right"]),
        key=str(payload["key"]),
        right_key=str(payload.get("rightKey") or payload["key"]),
        score=float(payload.get("score", 1.0)),
        join_type=str(payload.get("joinType", "equi")),
    )


def graph_to_json(graph: Dict[str, List[EdgeInfo]], tables: Dict[str, pd.DataFrame]) -> dict:
    node_names = sorted(set(tables.keys()) | set(graph.keys()))
    nodes = [
        {
            "id": name,
            "label": "Query" if name == "query" else name,
            "rowCount": int(tables[name].shape[0]) if name in tables else 0,
            "columns": [str(c) for c in tables[name].columns] if name in tables else [],
        }
        for name in node_names
    ]

    edge_map = {}
    for edges in graph.values():
        for edge in edges:
            pair = tuple(sorted([edge.left, edge.right]))
            join_keys = tuple(sorted([edge.key, edge.right_key or edge.key]))
            key = (pair[0], pair[1], join_keys[0], join_keys[1], edge.join_type)
            if key not in edge_map or edge.score > edge_map[key].score:
                edge_map[key] = edge

    return {
        "nodes": nodes,
        "edges": [
            {"id": f"{edge.left}-{edge.right}-{i}", **edge_to_json(edge)}
            for i, edge in enumerate(sorted(edge_map.values(), key=lambda e: (e.left, e.right, e.key)))
        ],
    }


def table_summaries(tables: Dict[str, pd.DataFrame]) -> list[dict]:
    return [
        {
            "name": name,
            "rows": int(df.shape[0]),
            "columns": [str(c) for c in df.columns],
            "columnCount": int(df.shape[1]),
        }
        for name, df in sorted(tables.items())
    ]


def path_to_json(
    idx: int,
    nodes: Sequence[str],
    score: float,
    edges: Sequence[EdgeInfo],
    tables: Dict[str, pd.DataFrame],
) -> dict:
    tuple_count = 0
    column_count = 0
    columns: list[str] = []
    try:
        from .paths import materialize_path

        mat = materialize_path(tables, nodes, edges)
        tuple_count = int(mat.shape[0])
        column_count = int(mat.shape[1])
        columns = [str(c) for c in mat.columns]
    except Exception:
        pass

    return {
        "id": f"path-{idx}",
        "rank": idx + 1,
        "nodes": list(nodes),
        "score": float(score),
        "edges": [edge_to_json(edge) for edge in edges],
        "nodeCount": len(nodes),
        "hopCount": max(0, len(nodes) - 1),
        "tupleCount": tuple_count,
        "columnCount": column_count,
        "columns": columns,
        "dot": path_to_dot(nodes, edges, tables),
    }


def paths_to_json(paths: Iterable[tuple[list[str], float, list[EdgeInfo]]], tables: Dict[str, pd.DataFrame]) -> list[dict]:
    return [path_to_json(i, nodes, score, edges, tables) for i, (nodes, score, edges) in enumerate(paths)]
