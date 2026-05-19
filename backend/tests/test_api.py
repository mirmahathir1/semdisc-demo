from __future__ import annotations


def test_health_endpoint(client):
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_demo_preprocess_graph_query_and_materialize_workflow(client):
    options = client.get("/api/datalakes")
    assert "orders" in options.get_json()["datalakes"]

    demo = client.get("/api/datalakes/orders")
    assert demo.status_code == 200
    dataset_id = demo.get_json()["datasetId"]

    preprocess = client.post(
        "/api/preprocess",
        json={"datasetId": dataset_id, "hyperparameters": {"joinEdgeThreshold": 0.1, "simplePathMaxLength": 4}},
    )
    assert preprocess.status_code == 200
    assert preprocess.get_json()["status"] == "ready"

    graph = client.get(f"/api/datasets/{dataset_id}/graph")
    assert graph.status_code == 200
    assert graph.get_json()["graph"]["edges"]

    # orders columns are text, so they join semantically (country/carrier/...).
    query = {
        "columns": ["country", "carrier"],
        "rows": [
            {"country": "US", "carrier": "DHL"},
            {"country": "SG", "carrier": "UPS"},
            {"country": "US", "carrier": "FedEx"},
            {"country": "XX", "carrier": "ZZ"},
        ],
    }
    submit = client.post("/api/query", json={"datasetId": dataset_id, "query": query})
    assert submit.status_code == 201
    query_id = submit.get_json()["queryId"]

    paths = client.get(f"/api/query/{query_id}/paths?k=5&maxHops=4")
    assert paths.status_code == 200
    first_path = paths.get_json()["paths"][0]
    assert "query" not in first_path["nodes"]
    assert set(first_path["nodes"]) == {"customers", "orders", "shipments"}
    assert {"country", "carrier"}.issubset(set(first_path["columns"]))
    assert first_path["queryTupleCoverage"] == "some"
    assert first_path["queryTupleCount"] == 4
    assert first_path["matchedQueryTupleCount"] == 3

    materialized = client.post(
        "/api/materialize",
        json={
            "datasetId": dataset_id,
            "nodes": first_path["nodes"],
            "edges": first_path["edges"],
            "pageSize": 5,
        },
    )
    assert materialized.status_code == 200
    assert materialized.get_json()["data"]["rows"]


def test_demo_graph_has_numeric_equi_and_semantic_joins(client):
    dataset_id = client.get("/api/demo/datalake").get_json()["datasetId"]
    client.post(
        "/api/preprocess",
        json={"datasetId": dataset_id, "hyperparameters": {"joinEdgeThreshold": 0.1}},
    )

    edges = client.get(f"/api/datasets/{dataset_id}/graph").get_json()["graph"]["edges"]
    counts = {}
    for edge in edges:
        counts[edge["joinType"]] = counts.get(edge["joinType"], 0) + 1

    # Numeric surrogate keys (customer_no/product_no/category_no/order_no)
    # produce equi-joins; text business keys join semantically.
    assert counts.get("equi", 0) >= 1
    assert counts.get("semantic", 0) >= 1

    def pairs(join_type):
        return {
            frozenset({(e["left"], e["key"]), (e["right"], e["rightKey"])})
            for e in edges
            if e["joinType"] == join_type
        }

    equi = pairs("equi")
    assert frozenset({("customers", "customer_no"), ("orders", "customer_no")}) in equi
    assert frozenset({("products", "product_no"), ("order_items", "product_no")}) in equi

    semantic = pairs("semantic")
    assert any(
        pair in semantic
        for pair in [
            frozenset({("customers", "customer_id"), ("orders", "customer_id")}),
            frozenset({("customers", "country"), ("orders", "ship_country")}),
        ]
    )
    assert any(
        pair in semantic
        for pair in [
            frozenset({("products", "product_id"), ("order_items", "product_id")}),
            frozenset({("products", "product"), ("order_items", "item_label")}),
        ]
    )
    assert any(
        pair in semantic
        for pair in [
            frozenset({("orders", "preferred_carrier"), ("shipments", "carrier")}),
            frozenset({("orders", "order_id"), ("shipments", "order_id")}),
        ]
    )


def test_semantic_type_threshold_filters_edges(client):
    dataset_id = client.get("/api/demo/datalake").get_json()["datasetId"]
    # Threshold above the max achievable similarity (exact match == 1.0)
    # must drop every semantic edge. Numeric equi-joins are unaffected
    # because they are not gated by semantic-type similarity.
    client.post(
        "/api/preprocess",
        json={
            "datasetId": dataset_id,
            "hyperparameters": {"joinEdgeThreshold": 0.1, "semanticTypeSimilarityThreshold": 1.5},
        },
    )
    edges = client.get(f"/api/datasets/{dataset_id}/graph").get_json()["graph"]["edges"]
    assert all(e["joinType"] != "semantic" for e in edges)
    assert any(e["joinType"] == "equi" for e in edges)

    # A reachable threshold restores them.
    client.post(
        "/api/preprocess",
        json={
            "datasetId": dataset_id,
            "hyperparameters": {"joinEdgeThreshold": 0.1, "semanticTypeSimilarityThreshold": 0.7},
        },
    )
    edges = client.get(f"/api/datasets/{dataset_id}/graph").get_json()["graph"]["edges"]
    assert any(e["joinType"] == "semantic" for e in edges)


def test_bundled_datalake_columns_have_semantic_types(client):
    datalakes = client.get("/api/datalakes").get_json()["datalakes"]

    for name in datalakes:
        dataset_id = client.get(f"/api/datalakes/{name}").get_json()["datasetId"]
        payload = client.get(f"/api/datasets/{dataset_id}/semantic-types").get_json()
        unknown = [
            f"{table}.{item['column']}"
            for table, items in payload["semanticTypes"].items()
            for item in items
            if item["semanticType"] == "Unknown"
        ]

        assert unknown == []


def test_unknown_dataset_returns_json_error(client):
    response = client.get("/api/datasets/missing/tables")

    assert response.status_code == 404
    assert response.get_json()["error"]["code"] == "not_found"
