from __future__ import annotations

import pandas as pd

from semdisc.join_graph import build_join_graph
from semdisc.paths import materialize_path, simple_paths_topk


def test_simple_paths_topk_returns_multihop_path():
    # Numeric keys so equi-joins form (text columns join semantically).
    tables = {
        "orders": pd.DataFrame({"order_id": [1], "product_id": [1]}),
        "items": pd.DataFrame({"order_id": [1], "product_id": [1]}),
        "products": pd.DataFrame({"product_id": [1]}),
    }
    graph = build_join_graph(tables, min_edge_score=0.8)

    paths = simple_paths_topk(graph, start="orders", max_hops=2, k=5)

    assert any(nodes == ["orders", "items", "products"] for nodes, _, _ in paths)


def test_materialize_path_merges_across_edges():
    tables = {
        "orders": pd.DataFrame({"order_id": [1], "customer_id": ["C1"]}),
        "shipments": pd.DataFrame({"order_id": [1], "carrier": ["DHL"]}),
    }
    graph = build_join_graph(tables, min_edge_score=0.8)
    edge = graph["orders"][0]

    materialized = materialize_path(tables, ["orders", "shipments"], [edge])

    assert materialized.iloc[0]["carrier"] == "DHL"

