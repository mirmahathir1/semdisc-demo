from __future__ import annotations

import pandas as pd

from semdisc.join_graph import SEMANTIC_TYPES, best_join_key_and_score, build_join_graph, jaccard
from semdisc.paths import simple_paths_topk
from semdisc.semantic_minhash import compute_column_minhashes


def test_jaccard_ignores_blank_values():
    assert jaccard(["A", "", "B"], ["B", "C", ""]) == 1 / 3


def test_best_join_key_uses_highest_overlap_numeric_column():
    # Equi-joins only form between numeric columns.
    left = pd.DataFrame({"id": [1, 2], "weak": [10, 20]})
    right = pd.DataFrame({"id": [1, 2], "weak": [30, 20]})

    minhashes = compute_column_minhashes({"left": left, "right": right})
    key, score = best_join_key_and_score("left", left, "right", right, minhashes)

    # Identical numeric set -> MinHash Jaccard exactly 1.0; "weak" overlaps
    # only partially, so "id" wins.
    assert key == "id"
    assert score == 1.0


def test_best_join_key_randomly_breaks_equal_numeric_scores(monkeypatch):
    left = pd.DataFrame({"id": [1, 2], "account_no": [10, 20]})
    right = pd.DataFrame({"id": [1, 2], "account_no": [10, 20]})

    minhashes = compute_column_minhashes({"left": left, "right": right})
    monkeypatch.setattr("semdisc.join_graph.random.choice", lambda candidates: candidates[-1])
    key, score = best_join_key_and_score("left", left, "right", right, minhashes)

    assert key == "account_no"
    assert score == 1.0


def test_text_common_columns_do_not_form_equi_joins():
    # order_id is text on both sides: no equi edge (text joins are semantic).
    left = pd.DataFrame({"order_id": ["O1", "O2"]})
    right = pd.DataFrame({"order_id": ["O1", "O2"]})

    minhashes = compute_column_minhashes({"left": left, "right": right})
    assert best_join_key_and_score("left", left, "right", right, minhashes) is None


def test_build_join_graph_creates_bidirectional_numeric_edges():
    tables = {
        "orders": pd.DataFrame({"order_id": [1, 2], "customer_id": ["C1", "C2"]}),
        "shipments": pd.DataFrame({"order_id": [1, 2], "shipment_id": ["S1", "S2"]}),
    }

    graph = build_join_graph(tables, min_edge_score=0.8)

    assert graph["orders"][0].right == "shipments"
    assert graph["shipments"][0].right == "orders"
    assert graph["orders"][0].key == "order_id"
    assert graph["orders"][0].join_type == "equi"


def test_equal_score_equi_and_semantic_edges_do_not_force_equi_first(monkeypatch):
    tables = {
        "left": pd.DataFrame({"business_id": ["A", "B"], "record_no": [1, 2]}),
        "right": pd.DataFrame({"business_id": ["A", "B"], "record_no": [1, 2]}),
    }
    monkeypatch.setitem(SEMANTIC_TYPES, ("left", "business_id"), "Business Identifier")
    monkeypatch.setitem(SEMANTIC_TYPES, ("right", "business_id"), "Business Identifier")
    monkeypatch.setattr("semdisc.join_graph.random.shuffle", lambda candidates: candidates.reverse())

    graph = build_join_graph(tables, min_edge_score=0.8)
    paths = simple_paths_topk(graph, start="left", max_hops=1, k=1)

    assert graph["left"][0].join_type == "semantic"
    assert paths[0][2][0].join_type == "semantic"
