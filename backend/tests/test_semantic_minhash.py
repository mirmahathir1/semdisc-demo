from __future__ import annotations

import pandas as pd

from semdisc.semantic_minhash import (
    compute_column_minhashes,
    jaccard,
    semantic_type_cosine,
)
from semdisc.join_graph import semantic_type_similarity


def test_semantic_type_similarity_falls_back_to_exact_match_without_sbert():
    # Tests run on the raw backend: no SBERT, so cosine is unavailable and
    # similarity degenerates to exact type-string match.
    assert semantic_type_cosine("Customer Identifier", "Order Identifier") is None
    assert semantic_type_similarity("Order Identifier", "Order Identifier") == 1.0
    assert semantic_type_similarity("Order Identifier", "Carrier Name") == 0.0
    assert semantic_type_similarity("", "") == 0.0


def test_minhash_jaccard_reflects_value_overlap():
    tables = {
        "a": pd.DataFrame({"k": ["x", "y", "z"], "empty": [None, None, None]}),
        "b": pd.DataFrame({"k": ["x", "y", "z"], "other": ["p", "q", "r"]}),
    }

    mh = compute_column_minhashes(tables, simhash_size=16, num_perm=128)

    # Identical value sets -> Jaccard exactly 1.0.
    assert jaccard(mh["a"]["k"].minhash, mh["b"]["k"].minhash) == 1.0
    # Disjoint value sets -> ~0.
    assert jaccard(mh["a"]["k"].minhash, mh["b"]["other"].minhash) < 0.1
    # Empty column -> None signature -> 0.0 (no spurious edge).
    assert mh["a"]["empty"].minhash is None
    assert jaccard(mh["a"]["empty"].minhash, mh["b"]["k"].minhash) == 0.0


def test_numeric_and_text_columns_are_classified():
    tables = {
        "t": pd.DataFrame(
            {
                "amount": [10, 20, 30],
                "year_str": ["2020", "2021", "2022"],
                "label": ["apple", "banana", "cherry"],
            }
        )
    }

    mh = compute_column_minhashes(tables)

    assert mh["t"]["amount"].is_numeric is True
    # >50% parse as numbers -> numeric even when stored as strings.
    assert mh["t"]["year_str"].is_numeric is True
    assert mh["t"]["label"].is_numeric is False


def test_compute_column_minhashes_is_cached():
    tables = {"t": pd.DataFrame({"c": ["1", "2"]})}

    first = compute_column_minhashes(tables, num_perm=64)
    second = compute_column_minhashes(tables, num_perm=64)

    # Same fingerprint -> same cached object returned.
    assert first is second
