from __future__ import annotations

import itertools
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .models import EdgeInfo
from .semantic_minhash import compute_column_minhashes, semantic_type_cosine
from .semantic_minhash import jaccard as minhash_jaccard


_DATA_DIR = Path(__file__).resolve().parent / "data"


def _load_bundled_semantic_types() -> Dict[Tuple[str, str], str]:
    """Load semantic types from every bundled data lake's ``semantic_types.json``.

    Each data lake caches its column semantic types beside its CSVs (see
    :mod:`semdisc.semantic_type_extractor`). Table names are unique across the
    bundled lakes, so they share one ``(table, column) -> semantic_type`` map.
    Missing or unreadable caches are skipped; services may extract and merge
    types for a data lake at request time via :func:`register_semantic_types`.
    """
    # Imported lazily to avoid a circular import (data_loader has no deps here).
    from .data_loader import DATALAKE_DIR, list_bundled_datalakes
    from .semantic_type_extractor import SEMANTIC_TYPES_FILENAME, load_semantic_types

    merged: Dict[Tuple[str, str], str] = {}
    try:
        lakes = list_bundled_datalakes()
    except FileNotFoundError:
        return merged
    for lake in lakes:
        cache_path = DATALAKE_DIR / lake / SEMANTIC_TYPES_FILENAME
        loaded = load_semantic_types(cache_path)
        if loaded:
            merged.update(loaded)
    return merged


def register_semantic_types(semantic_types: Dict[Tuple[str, str], str]) -> None:
    """Merge ``(table, column) -> semantic_type`` entries into the global map.

    Used when a dataset's semantic types are loaded or freshly extracted so the
    join builder (which reads :data:`SEMANTIC_TYPES`) can see them.
    """
    SEMANTIC_TYPES.update(semantic_types)


# Populated from the bundled data lakes' JSON caches at import time. Kept as a
# mutable module-level dict so freshly extracted types can be merged in.
SEMANTIC_TYPES: Dict[Tuple[str, str], str] = _load_bundled_semantic_types()


def _load_query_semantic_column_map() -> Dict[str, str]:
    """Map query-header aliases to canonical column names (from JSON config)."""
    path = _DATA_DIR / "query_column_map.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# Aliases a query column header may use, mapped to the canonical column name
# used in the data lakes. Sourced from ``data/query_column_map.json``.
QUERY_SEMANTIC_COLUMN_MAP = _load_query_semantic_column_map()


def normalize_cell(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    """Exact set Jaccard over raw values. Kept for reference/tests; edge
    weights now come from MinHash signatures (see ``semantic_minhash``)."""
    sa = {x for x in a if x}
    sb = {x for x in b if x}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / float(len(sa | sb))


def best_join_key_and_score(
    left_name: str,
    left: pd.DataFrame,
    right_name: str,
    right: pd.DataFrame,
    column_minhashes: Dict[str, Dict[str, object]],
) -> Optional[Tuple[str, float]]:
    right_cols = set(map(str, right.columns))
    common_cols = [str(c) for c in left.columns if str(c) in right_cols]
    if not common_cols:
        return None

    best_score: Optional[float] = None
    best: List[Tuple[str, float]] = []
    for col in common_cols:
        left_sig = column_minhashes.get(left_name, {}).get(col)
        right_sig = column_minhashes.get(right_name, {}).get(col)
        # Equi-joins are only formed between numeric columns; text columns
        # join semantically instead.
        if left_sig is None or right_sig is None:
            continue
        if not (left_sig.is_numeric and right_sig.is_numeric):
            continue
        score = minhash_jaccard(left_sig.minhash, right_sig.minhash)
        if best_score is None or score > best_score:
            best_score = score
            best = [(col, float(score))]
        elif score == best_score:
            best.append((col, float(score)))
    return random.choice(best) if best else None


def semantic_type_similarity(type_a: str, type_b: str) -> float:
    """Similarity between two semantic-type strings. Uses the cosine of their
    SBERT embeddings when available; otherwise falls back to exact string
    match (1.0 if identical, else 0.0) — the degenerate case of cosine."""
    cos = semantic_type_cosine(type_a, type_b)
    if cos is not None:
        return cos
    return 1.0 if (type_a or "").strip() == (type_b or "").strip() and type_a else 0.0


def best_semantic_join_keys_and_score(
    left_name: str,
    left: pd.DataFrame,
    right_name: str,
    right: pd.DataFrame,
    column_minhashes: Dict[str, Dict[str, object]],
    semantic_type_threshold: float = 0.7,
) -> Optional[Tuple[str, str, float]]:
    best_score: Optional[float] = None
    best: List[Tuple[str, str, float]] = []

    for lc in map(str, left.columns):
        left_type = SEMANTIC_TYPES.get((left_name, lc))
        if not left_type:
            continue

        for rc in map(str, right.columns):
            right_type = SEMANTIC_TYPES.get((right_name, rc))
            if not right_type:
                continue

            # Only pair columns whose semantic-type embeddings are at least
            # `semantic_type_threshold` cosine-similar.
            if semantic_type_similarity(left_type, right_type) < semantic_type_threshold:
                continue

            left_sig = column_minhashes.get(left_name, {}).get(lc)
            right_sig = column_minhashes.get(right_name, {}).get(rc)
            # Semantic joins are only formed between text columns; numeric
            # columns join via equi-joins instead.
            if left_sig is None or right_sig is None:
                continue
            if left_sig.is_numeric or right_sig.is_numeric:
                continue
            score = minhash_jaccard(left_sig.minhash, right_sig.minhash)
            if best_score is None or score > best_score:
                best_score = score
                best = [(lc, rc, float(score))]
            elif score == best_score:
                best.append((lc, rc, float(score)))

    return random.choice(best) if best else None


def _append_best_edge_pair(
    graph: Dict[str, List[EdgeInfo]],
    edge_pairs: List[Tuple[EdgeInfo, EdgeInfo]],
) -> None:
    if not edge_pairs:
        return

    best_score = max(forward.score for forward, _ in edge_pairs)
    tied = [edge_pair for edge_pair in edge_pairs if edge_pair[0].score == best_score]
    forward, reverse = random.choice(tied)
    graph[forward.left].append(forward)
    graph[reverse.left].append(reverse)


def build_join_graph(
    tables: Dict[str, pd.DataFrame],
    min_edge_score: float,
    *,
    simhash_size: int = 32,
    num_perm: int = 128,
    semantic_type_threshold: float = 0.7,
    column_minhashes: Optional[Dict[str, Dict[str, object]]] = None,
) -> Dict[str, List[EdgeInfo]]:
    graph: Dict[str, List[EdgeInfo]] = defaultdict(list)
    names = sorted(tables.keys())

    if column_minhashes is None:
        column_minhashes = compute_column_minhashes(
            tables, simhash_size=simhash_size, num_perm=num_perm
        )

    for a, b in itertools.combinations(names, 2):
        edge_pairs: List[Tuple[EdgeInfo, EdgeInfo]] = []

        best_equi = best_join_key_and_score(a, tables[a], b, tables[b], column_minhashes)
        if best_equi is not None:
            key, score = best_equi
            if score >= min_edge_score:
                edge_pairs.append(
                    (
                        EdgeInfo(a, b, key, score, key, "equi"),
                        EdgeInfo(b, a, key, score, key, "equi"),
                    )
                )

        best_sem = best_semantic_join_keys_and_score(
            a, tables[a], b, tables[b], column_minhashes, semantic_type_threshold
        )
        if best_sem is not None:
            left_key, right_key, score = best_sem
            if score >= min_edge_score:
                edge_pairs.append(
                    (
                        EdgeInfo(a, b, left_key, score, right_key, "semantic"),
                        EdgeInfo(b, a, right_key, score, left_key, "semantic"),
                    )
                )

        _append_best_edge_pair(graph, edge_pairs)

    return graph
