from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import pandas as pd

from .data_loader import (
    UPLOADED_SEMANTIC_TYPES_DIR,
    bundled_datalake,
    bundled_query,
    bundled_semantic_types_path,
    ensure_nonempty_tables,
    list_bundled_datalakes,
)
from .join_graph import (
    QUERY_SEMANTIC_COLUMN_MAP,
    SEMANTIC_TYPES,
    build_join_graph,
    normalize_cell,
    register_semantic_types,
)
from .models import EdgeInfo
from .semantic_type_extractor import load_or_extract_semantic_types
from .paths import PathTuple, materialize_path, simple_paths_topk
from .serialization import (
    dataframe_to_json,
    edge_from_json,
    graph_to_json,
    paths_to_json,
    table_summaries,
)


class ApiError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


@dataclass
class DatasetState:
    id: str
    tables: Dict[str, pd.DataFrame]
    source: str = "uploaded"
    graph: Dict[str, List[EdgeInfo]] = field(default_factory=dict)
    preprocessed: bool = False
    hyperparameters: dict = field(default_factory=dict)
    # Where this dataset's semantic_types.json cache lives (set at creation).
    semantic_types_path: Optional[Path] = None
    semantic_types_loaded: bool = False


@dataclass
class QueryState:
    id: str
    dataset_id: str
    query: pd.DataFrame


DATASETS: Dict[str, DatasetState] = {}
QUERIES: Dict[str, QueryState] = {}


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _dataset_or_404(dataset_id: str) -> DatasetState:
    try:
        return DATASETS[dataset_id]
    except KeyError as exc:
        raise ApiError("not_found", f"Dataset '{dataset_id}' was not found.", 404) from exc


def _query_or_404(query_id: str) -> QueryState:
    try:
        return QUERIES[query_id]
    except KeyError as exc:
        raise ApiError("not_found", f"Query '{query_id}' was not found.", 404) from exc


def create_dataset(
    tables: Dict[str, pd.DataFrame],
    *,
    source: str = "uploaded",
    semantic_types_path: Optional[Path] = None,
) -> DatasetState:
    clean = ensure_nonempty_tables(tables)
    if not clean:
        raise ApiError("empty_dataset", "No non-empty CSV tables were provided.", 400)
    state = DatasetState(id=_new_id("ds"), tables=clean, source=source)
    # Uploaded datasets cache their extracted semantic types under a stable
    # per-dataset path so a re-run reuses the JSON instead of calling the LLM.
    state.semantic_types_path = (
        semantic_types_path or UPLOADED_SEMANTIC_TYPES_DIR / f"{state.id}.json"
    )
    DATASETS[state.id] = state
    return state


def create_demo_dataset() -> DatasetState:
    return create_bundled_dataset()


def create_bundled_dataset(name: str | None = None) -> DatasetState:
    source = f"bundled:{name}" if name else "bundled"
    return create_dataset(
        bundled_datalake(name),
        source=source,
        semantic_types_path=bundled_semantic_types_path(name),
    )


def ensure_semantic_types(state: DatasetState) -> DatasetState:
    """Load (or, on cache miss, extract via LLM) this dataset's semantic types
    and merge them into the global ``SEMANTIC_TYPES`` map used by the join
    builder. Runs at most once per dataset."""
    if state.semantic_types_loaded:
        return state
    cache_path = state.semantic_types_path or (
        UPLOADED_SEMANTIC_TYPES_DIR / f"{state.id}.json"
    )
    semantic_types = load_or_extract_semantic_types(state.tables, cache_path)
    register_semantic_types(semantic_types)
    state.semantic_types_loaded = True
    return state


def preprocess_dataset(dataset_id: str, hyperparameters: Optional[dict] = None) -> DatasetState:
    state = ensure_semantic_types(_dataset_or_404(dataset_id))
    hyperparameters = dict(hyperparameters or {})
    min_edge_score = float(hyperparameters.get("joinEdgeThreshold", hyperparameters.get("minEdgeScore", 0.1)))
    simhash_size = int(hyperparameters.get("simhashSize", 32))
    num_perm = int(hyperparameters.get("numHashesMinhash", 128))
    semantic_type_threshold = float(hyperparameters.get("semanticTypeSimilarityThreshold", 0.7))
    state.graph = build_join_graph(
        state.tables,
        min_edge_score=min_edge_score,
        simhash_size=simhash_size,
        num_perm=num_perm,
        semantic_type_threshold=semantic_type_threshold,
    )
    state.preprocessed = True
    state.hyperparameters = hyperparameters
    return state


def ensure_preprocessed(state: DatasetState) -> DatasetState:
    if not state.preprocessed:
        preprocess_dataset(state.id, {"joinEdgeThreshold": 0.1, "simplePathMaxLength": 4})
    return state


def dataset_payload(state: DatasetState) -> dict:
    return {
        "datasetId": state.id,
        "source": state.source,
        "preprocessed": state.preprocessed,
        "tables": table_summaries(state.tables),
    }


def demo_datalake_payload() -> dict:
    state = create_bundled_dataset()
    return dataset_payload(state)


def bundled_datalakes_payload() -> dict:
    return {"datalakes": list_bundled_datalakes()}


def bundled_datalake_payload(name: str) -> dict:
    state = create_bundled_dataset(name)
    payload = dataset_payload(state)
    payload["datalake"] = name
    return payload


def demo_query_payload(name: str | None = None) -> dict:
    return {"query": dataframe_to_json(bundled_query(name), page_size=1000)}


def table_preview(dataset_id: str, table_name: str, *, page: int = 1, page_size: int = 100) -> dict:
    state = _dataset_or_404(dataset_id)
    if table_name not in state.tables:
        raise ApiError("not_found", f"Table '{table_name}' was not found.", 404)
    return {"table": table_name, "data": dataframe_to_json(state.tables[table_name], page=page, page_size=page_size)}


def semantic_types_payload(dataset_id: str) -> dict:
    state = ensure_semantic_types(_dataset_or_404(dataset_id))
    payload = {}
    for table_name, df in sorted(state.tables.items()):
        payload[table_name] = [
            {
                "column": str(col),
                "semanticType": SEMANTIC_TYPES.get((table_name, str(col)), "Unknown"),
            }
            for col in df.columns
        ]
    return {"datasetId": dataset_id, "semanticTypes": payload}


def graph_payload(dataset_id: str, *, min_edge_score: Optional[float] = None) -> dict:
    state = ensure_preprocessed(_dataset_or_404(dataset_id))
    graph = state.graph
    if min_edge_score is not None:
        graph = build_join_graph(
            state.tables,
            float(min_edge_score),
            simhash_size=int(state.hyperparameters.get("simhashSize", 32)),
            num_perm=int(state.hyperparameters.get("numHashesMinhash", 128)),
            semantic_type_threshold=float(
                state.hyperparameters.get("semanticTypeSimilarityThreshold", 0.7)
            ),
        )
    return {"datasetId": dataset_id, "graph": graph_to_json(graph, state.tables)}


def navigation_paths_payload(dataset_id: str, *, max_hops: int = 4, k: int = 10) -> dict:
    state = ensure_preprocessed(_dataset_or_404(dataset_id))
    by_nodes: Dict[tuple[str, ...], PathTuple] = {}
    for start in sorted(state.tables.keys()):
        for nodes, score, edges in simple_paths_topk(state.graph, start=start, max_hops=max_hops, k=k):
            if len(nodes) <= 1:
                continue
            key = tuple(nodes)
            prev = by_nodes.get(key)
            if prev is None or score > prev[1]:
                by_nodes[key] = (nodes, score, edges)
    paths = sorted(by_nodes.values(), key=lambda x: x[1], reverse=True)
    return {"datasetId": dataset_id, "paths": paths_to_json(paths, state.tables)}


def _df_from_payload(payload: dict) -> pd.DataFrame:
    if "query" in payload and isinstance(payload["query"], dict):
        payload = payload["query"]

    columns = payload.get("columns")
    rows = payload.get("rows")
    if rows is None:
        raise ApiError("bad_request", "Query payload must include rows.", 400)
    if rows and isinstance(rows[0], list):
        if not columns:
            raise ApiError("bad_request", "List-style rows require columns.", 400)
        return pd.DataFrame(rows, columns=columns)
    return pd.DataFrame(rows, columns=columns)


def submit_query(dataset_id: str, payload: dict) -> QueryState:
    state = ensure_preprocessed(_dataset_or_404(dataset_id))
    query_df = _df_from_payload(payload)
    if query_df.empty or query_df.shape[1] == 0:
        raise ApiError("empty_query", "Query table must include at least one row and one column.", 400)
    qstate = QueryState(
        id=_new_id("qry"),
        dataset_id=dataset_id,
        query=query_df,
    )
    QUERIES[qstate.id] = qstate
    return qstate


def _normalized_column(column: str) -> str:
    return str(column or "").strip().lower()


def _query_column_candidates(column: str) -> Set[str]:
    clean = str(column or "").strip()
    candidates = {clean}
    mapped = QUERY_SEMANTIC_COLUMN_MAP.get(clean)
    if mapped:
        candidates.add(mapped)
    return {_normalized_column(candidate) for candidate in candidates if candidate}


def _covered_query_columns(nodes: Sequence[str], tables: Dict[str, pd.DataFrame], query: pd.DataFrame) -> Set[str]:
    path_columns = {
        _normalized_column(column)
        for node in nodes
        for column in (tables[node].columns if node in tables else [])
    }
    covered = set()
    for query_column in map(str, query.columns):
        if path_columns.intersection(_query_column_candidates(query_column)):
            covered.add(query_column)
    return covered


def query_relevant_paths(
    graph: Dict[str, List[EdgeInfo]],
    tables: Dict[str, pd.DataFrame],
    query: pd.DataFrame,
    *,
    max_hops: int,
    k: int,
) -> List[PathTuple]:
    by_nodes: Dict[tuple[str, ...], tuple[int, float, PathTuple]] = {}
    per_start_limit = max(50, k * max(4, max_hops))
    query_columns = [str(column) for column in query.columns]

    for start in sorted(tables.keys()):
        for nodes, score, edges in simple_paths_topk(graph, start=start, max_hops=max_hops, k=per_start_limit):
            if len(nodes) <= 1:
                continue
            covered = _covered_query_columns(nodes, tables, query)
            if not covered:
                continue
            item: PathTuple = (nodes, score, edges)
            key = tuple(nodes)
            ranking = (len(covered), score)
            prev = by_nodes.get(key)
            if prev is None or ranking > (prev[0], prev[1]):
                by_nodes[key] = (len(covered), score, item)

    if not by_nodes:
        return []

    target_count = len(query_columns)
    scored = list(by_nodes.values())
    complete = [item for item in scored if item[0] == target_count]
    pool = complete or scored
    pool.sort(key=lambda item: (item[0], -len(item[2][0]), item[1]), reverse=True)
    return [item[2] for item in pool[:k]]


def _query_tuple_coverage(
    nodes: Sequence[str],
    edges: Sequence[EdgeInfo],
    tables: Dict[str, pd.DataFrame],
    query: pd.DataFrame,
) -> dict:
    total = int(query.shape[0])
    if total == 0:
        return {
            "queryTupleCount": 0,
            "matchedQueryTupleCount": 0,
            "matchedQueryRowIndices": [],
            "queryTupleCoverage": "none",
        }

    try:
        materialized = materialize_path(tables, nodes, edges)
    except Exception:
        return {
            "queryTupleCount": total,
            "matchedQueryTupleCount": 0,
            "matchedQueryRowIndices": [],
            "queryTupleCoverage": "none",
        }

    materialized_columns = {
        _normalized_column(column): str(column)
        for column in materialized.columns
    }
    column_pairs = []
    for query_column in map(str, query.columns):
        candidates = _query_column_candidates(query_column)
        materialized_column = next(
            (materialized_columns[candidate] for candidate in candidates if candidate in materialized_columns),
            None,
        )
        if materialized_column is None:
            return {
                "queryTupleCount": total,
                "matchedQueryTupleCount": 0,
                "matchedQueryRowIndices": [],
                "queryTupleCoverage": "none",
            }
        column_pairs.append((query_column, materialized_column))

    matched_indices = []
    normalized_materialized = {
        column: materialized[column].map(normalize_cell)
        for _, column in column_pairs
    }
    for row_index, (_, query_row) in enumerate(query.iterrows()):
        row_matches = pd.Series(True, index=materialized.index)
        for query_column, materialized_column in column_pairs:
            row_matches &= normalized_materialized[materialized_column] == normalize_cell(query_row[query_column])
        if bool(row_matches.any()):
            matched_indices.append(row_index)

    matched = len(matched_indices)
    if matched == total:
        coverage = "all"
    elif matched > 0:
        coverage = "some"
    else:
        coverage = "none"

    return {
        "queryTupleCount": total,
        "matchedQueryTupleCount": matched,
        "matchedQueryRowIndices": matched_indices,
        "queryTupleCoverage": coverage,
    }


def query_paths_payload(query_id: str, *, max_hops: int = 4, k: int = 10) -> dict:
    qstate = _query_or_404(query_id)
    state = ensure_preprocessed(_dataset_or_404(qstate.dataset_id))
    paths = query_relevant_paths(state.graph, state.tables, qstate.query, max_hops=max_hops, k=k)
    serialized = paths_to_json(paths, state.tables)
    for payload, (nodes, _, edges) in zip(serialized, paths):
        payload.update(_query_tuple_coverage(nodes, edges, state.tables, qstate.query))
    return {"queryId": query_id, "datasetId": qstate.dataset_id, "paths": serialized}


def materialize_payload(payload: dict) -> dict:
    query_id = payload.get("queryId")
    dataset_id = payload.get("datasetId")
    nodes = payload.get("nodes") or payload.get("pathNodes")
    edge_payloads = payload.get("edges") or []
    page = int(payload.get("page", 1))
    page_size = int(payload.get("pageSize", 100))

    if not nodes:
        raise ApiError("bad_request", "Materialization requires path nodes.", 400)

    if query_id:
        qstate = _query_or_404(str(query_id))
        tables = _dataset_or_404(qstate.dataset_id).tables
    elif dataset_id:
        tables = _dataset_or_404(str(dataset_id)).tables
    else:
        raise ApiError("bad_request", "Materialization requires datasetId or queryId.", 400)

    edges = [edge_from_json(edge) for edge in edge_payloads]
    materialized = materialize_path(tables, [str(n) for n in nodes], edges)
    return {"data": dataframe_to_json(materialized, page=page, page_size=page_size)}
