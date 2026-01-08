from __future__ import annotations

import io
import itertools
import json
import hashlib
import math
import time
import zipfile
from contextlib import nullcontext
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd
import streamlit as st


@dataclass(frozen=True)
class EdgeInfo:
    left: str
    right: str
    key: str
    score: float


# Hardcoded semantic type annotations (demo-only).
# Maps (table_name, column_name) -> semantic type string.
SEMANTIC_TYPES: Dict[Tuple[str, str], str] = {
    # customers
    ("customers", "customer_id"): "Customer Identifier",
    ("customers", "name"): "Customer Full Name",
    ("customers", "country"): "Customer Country",
    ("customers", "segment"): "Customer Segment",
    ("customers", "signup_date"): "Customer Signup Date",
    # orders
    ("orders", "order_id"): "Order Identifier",
    ("orders", "customer_id"): "Customer Identifier",
    ("orders", "order_date"): "Order Date",
    ("orders", "order_status"): "Order Status",
    ("orders", "currency"): "Currency Code",
    # order_items
    ("order_items", "order_id"): "Order Identifier",
    ("order_items", "product_id"): "Product Identifier",
    ("order_items", "qty"): "Quantity Ordered",
    ("order_items", "unit_price"): "Unit Price",
    # products
    ("products", "product_id"): "Product Identifier",
    ("products", "product"): "Product Name",
    ("products", "category_id"): "Product Category Identifier",
    ("products", "supplier_id"): "Supplier Identifier",
    ("products", "list_price"): "Product List Price",
    # categories
    ("categories", "category_id"): "Product Category Identifier",
    ("categories", "category"): "Product Category Name",
    ("categories", "department_id"): "Department Identifier",
    # departments
    ("departments", "department_id"): "Department Identifier",
    ("departments", "department"): "Department Name",
    # suppliers
    ("suppliers", "supplier_id"): "Supplier Identifier",
    ("suppliers", "supplier"): "Supplier Name",
    ("suppliers", "supplier_country"): "Supplier Country",
    # warehouses
    ("warehouses", "warehouse_id"): "Warehouse Identifier",
    ("warehouses", "warehouse_city"): "Warehouse City",
    ("warehouses", "warehouse_country"): "Warehouse Country",
    # inventory
    ("inventory", "warehouse_id"): "Warehouse Identifier",
    ("inventory", "product_id"): "Product Identifier",
    ("inventory", "on_hand_qty"): "On-hand Inventory Quantity",
    ("inventory", "reorder_point"): "Inventory Reorder Point",
    # shipments
    ("shipments", "shipment_id"): "Shipment Identifier",
    ("shipments", "order_id"): "Order Identifier",
    ("shipments", "warehouse_id"): "Warehouse Identifier",
    ("shipments", "carrier_id"): "Carrier Identifier",
    ("shipments", "ship_date"): "Shipment Date",
    # carriers
    ("carriers", "carrier_id"): "Carrier Identifier",
    ("carriers", "carrier"): "Carrier Name",
    # payments
    ("payments", "payment_id"): "Payment Identifier",
    ("payments", "order_id"): "Order Identifier",
    ("payments", "payment_method_id"): "Payment Method Identifier",
    ("payments", "amount"): "Payment Amount",
    ("payments", "payment_date"): "Payment Date",
    # payment_methods
    ("payment_methods", "payment_method_id"): "Payment Method Identifier",
    ("payment_methods", "payment_method"): "Payment Method Name",
    # returns
    ("returns", "return_id"): "Return Identifier",
    ("returns", "order_id"): "Order Identifier",
    ("returns", "product_id"): "Product Identifier",
    ("returns", "return_date"): "Return Date",
    ("returns", "return_reason"): "Return Reason",
    ("returns", "return_amount"): "Return Amount",
    # reviews
    ("reviews", "review_id"): "Review Identifier",
    ("reviews", "customer_id"): "Customer Identifier",
    ("reviews", "product_id"): "Product Identifier",
    ("reviews", "rating"): "Product Rating",
    ("reviews", "review_date"): "Review Date",
}


def normalize_cell(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa = {x for x in a if x}
    sb = {x for x in b if x}
    if not sa and not sb:
        return 0.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / float(len(sa | sb))


def best_join_key_and_score(left: pd.DataFrame, right: pd.DataFrame) -> Optional[Tuple[str, float]]:
    common_cols = [c for c in left.columns if c in right.columns]
    if not common_cols:
        return None

    best = None
    for col in common_cols:
        lvals = (normalize_cell(v) for v in left[col].head(5000))
        rvals = (normalize_cell(v) for v in right[col].head(5000))
        score = jaccard(lvals, rvals)
        if best is None or score > best[1]:
            best = (col, score)
    return best


def build_join_graph(tables: Dict[str, pd.DataFrame], min_edge_score: float) -> Dict[str, List[EdgeInfo]]:
    graph: Dict[str, List[EdgeInfo]] = defaultdict(list)
    names = sorted(tables.keys())
    for a, b in itertools.combinations(names, 2):
        best = best_join_key_and_score(tables[a], tables[b])
        if best is None:
            continue
        key, score = best
        if score < min_edge_score:
            continue
        graph[a].append(EdgeInfo(left=a, right=b, key=key, score=score))
        graph[b].append(EdgeInfo(left=b, right=a, key=key, score=score))
    return graph


def attach_query_node(
    graph: Dict[str, List[EdgeInfo]],
    tables: Dict[str, pd.DataFrame],
    query: pd.DataFrame,
    min_edge_score: float,
) -> Tuple[Dict[str, List[EdgeInfo]], Dict[str, pd.DataFrame]]:
    tables2 = dict(tables)
    tables2["query"] = query
    g2: Dict[str, List[EdgeInfo]] = defaultdict(list, {k: list(v) for k, v in graph.items()})

    for name, df in tables.items():
        best = best_join_key_and_score(query, df)
        if best is None:
            continue
        key, score = best
        if score < min_edge_score:
            continue
        g2["query"].append(EdgeInfo(left="query", right=name, key=key, score=score))
        g2[name].append(EdgeInfo(left=name, right="query", key=key, score=score))

    return g2, tables2


def simple_paths_topk(
    graph: Dict[str, List[EdgeInfo]],
    start: str,
    max_hops: int,
    k: int,
) -> List[Tuple[List[str], float, List[EdgeInfo]]]:
    # Demo ranking: prefer "good" edges, but do not collapse everything to 1-hop.
    #
    # Key idea: multiplicative scoring makes longer paths vanish (product shrinks),
    # so we instead use a length-normalized aggregation (geometric mean) and then
    # deliberately diversify the returned set to include some longer paths.
    results: List[Tuple[List[str], float, List[EdgeInfo]]] = []
    q = deque()
    q.append((start, [start], []))  # node, path, edges

    while q:
        node, path, edges = q.popleft()
        if len(path) > 1:
            hop_count = len(path) - 1

            # Length-normalized edge strength (0..1) using geometric mean.
            # This avoids auto-favoring 1-hop paths purely due to multiplication.
            eps = 1e-9
            log_sum = 0.0
            for e in edges:
                log_sum += math.log(max(e.score, eps))
            geom_mean = math.exp(log_sum / float(hop_count)) if hop_count > 0 else 0.0

            # Small length bonus so multi-hop paths can appear in the top-k list.
            length_bonus = 1.0 + 0.12 * hop_count
            adjusted = float(geom_mean * length_bonus)
            results.append((path, adjusted, edges))

        if len(path) - 1 >= max_hops:
            continue

        for e in graph.get(node, []):
            # Joins are non-directional: get the other node regardless of left/right
            if e.left == node:
                nxt = e.right
            elif e.right == node:
                nxt = e.left
            else:
                continue  # Edge doesn't connect to current node
            if nxt in path:
                continue
            q.append((nxt, path + [nxt], edges + [e]))

    # Diversify by hop count so the UI includes some 4–5 node join paths when they exist.
    by_hops: Dict[int, List[Tuple[List[str], float, List[EdgeInfo]]]] = defaultdict(list)
    for path, score, edges in results:
        by_hops[len(path) - 1].append((path, score, edges))
    for hop_count in by_hops:
        by_hops[hop_count].sort(key=lambda x: x[1], reverse=True)

    picked: List[Tuple[List[str], float, List[EdgeInfo]]] = []
    seen: Set[Tuple[str, ...]] = set()

    # Coverage-first: pick the best path for each hop count (1..max_hops) so the UI
    # naturally includes node counts 2..(max_hops+1) when available.
    for hop_count in range(1, max_hops + 1):
        items = by_hops.get(hop_count, [])
        if not items:
            continue
        path, score, edges = items[0]
        key = tuple(path)
        if key in seen:
            continue
        picked.append((path, score, edges))
        seen.add(key)
        if len(picked) >= k:
            return picked[:k]

    # Fill the rest best-first across all hop counts.
    leftovers: List[Tuple[List[str], float, List[EdgeInfo]]] = []
    for hop_count, items in by_hops.items():
        leftovers.extend(items)
    leftovers.sort(key=lambda x: x[1], reverse=True)
    for path, score, edges in leftovers:
        if len(picked) >= k:
            break
        key = tuple(path)
        if key in seen:
            continue
        picked.append((path, score, edges))
        seen.add(key)

    return picked[:k]


def diversify_paths_by_node_count(
    paths: Sequence[Tuple[List[str], float, List[EdgeInfo]]],
    *,
    min_nodes: int,
    max_nodes: int,
    k: int,
) -> List[Tuple[List[str], float, List[EdgeInfo]]]:
    """Pick up to k paths, ensuring a mix of node counts when possible.

    Intended for UI display: if there exists at least one path for each node count
    in [min_nodes, max_nodes], the returned list will include at least one of each.
    """

    by_n: Dict[int, List[Tuple[List[str], float, List[EdgeInfo]]]] = defaultdict(list)
    for nodes, score, edges in paths:
        n = len(nodes)
        if min_nodes <= n <= max_nodes:
            by_n[n].append((nodes, score, edges))

    picked: List[Tuple[List[str], float, List[EdgeInfo]]] = []
    seen: Set[Tuple[str, ...]] = set()

    # Sort each node-count group by score so round-robin still keeps "best-first"
    # within each length.
    for n in by_n:
        by_n[n].sort(key=lambda x: x[1], reverse=True)

    # Round-robin across node counts so we don't collapse into only max_nodes.
    idx_by_n: Dict[int, int] = {n: 0 for n in range(min_nodes, max_nodes + 1)}

    while len(picked) < k:
        progressed = False
        for n in range(min_nodes, max_nodes + 1):
            items = by_n.get(n, [])
            i = idx_by_n.get(n, 0)
            # Skip already-seen paths (can happen if caller already deduped partially).
            while i < len(items) and tuple(items[i][0]) in seen:
                i += 1
            idx_by_n[n] = i
            if i >= len(items):
                continue
            nodes, score, edges = items[i]
            idx_by_n[n] = i + 1
            key = tuple(nodes)
            if key in seen:
                continue
            picked.append((nodes, score, edges))
            seen.add(key)
            progressed = True
            if len(picked) >= k:
                break
        if not progressed:
            break

    return picked[:k]


def materialize_path(
    tables: Dict[str, pd.DataFrame],
    path_nodes: Sequence[str],
    path_edges: Sequence[EdgeInfo],
) -> pd.DataFrame:
    df = tables[path_nodes[0]].copy()
    for edge, next_node in zip(path_edges, path_nodes[1:]):
        # Joins are non-directional: merge with the next node regardless of edge direction
        right = tables[next_node]
        key = edge.key
        df = df.merge(right, how="left", on=key, suffixes=("", f"_{next_node}"))
    return df


def to_dot(
    graph: Dict[str, List[EdgeInfo]],
    highlight_edges: Sequence[Tuple[str, str]] = (),
    tables: Optional[Dict[str, pd.DataFrame]] = None,
) -> str:
    highlight = set()
    for a, b in highlight_edges:
        highlight.add((a, b))
        highlight.add((b, a))

    # Build a stable edge set to avoid double-printing undirected edges.
    undirected = {}
    for a, edges in graph.items():
        for e in edges:
            key = tuple(sorted([e.left, e.right]))
            undirected[key] = e

    lines = [
        "graph G {",
        # Tighter layout: reduce spacing between ranks/nodes for shorter edges.
        '  graph [rankdir="LR", bgcolor="white", ranksep="0.14", nodesep="0.10"];',
        # Make nodes more compact (smaller height) for long join paths.
        # Graphviz node `height` is a minimum (in inches) and labels can still expand it,
        # so we also reduce font size and internal margins.
        '  node [shape="box", style="rounded,filled", fillcolor="#f7f7f7", color="#444444", fontname="Helvetica", fontsize="7", margin="0.03,0.015", height="0.12", width="0.26"];',
        # Smaller edge label font so join-key labels don't dominate.
        '  edge [color="#666666", fontname="Helvetica", fontsize="7"];',
    ]

    for (u, v), e in sorted(undirected.items(), key=lambda kv: kv[0]):
        is_hl = (u, v) in highlight or (v, u) in highlight
        color = "#d62728" if is_hl else "#666666"
        penwidth = "2.0" if is_hl else "0.9"
        
        # Get the join key column name from each table
        # Currently e.key is the common column name, but we show it for both tables
        left_join_key = e.key
        right_join_key = e.key
        
        # Show just the two join keys separated by comma
        label = f"{left_join_key}, {right_join_key}"
        
        lines.append(f'  "{u}" -- "{v}" [label="{label}", color="{color}", penwidth="{penwidth}"];')

    lines.append("}")
    return "\n".join(lines)


@st.cache_data(show_spinner=False)
def bundled_datalake() -> Dict[str, pd.DataFrame]:
    # Expanded demo datalake (e-commerce domain). The join graph is inferred purely by
    # (1) shared column names and (2) overlapping values (Jaccard).
    customers = pd.DataFrame(
        [
            {"customer_id": "C001", "name": "A. Rivera", "country": "US", "segment": "Consumer", "signup_date": "2025-01-08"},
            {"customer_id": "C002", "name": "S. Tan", "country": "SG", "segment": "SMB", "signup_date": "2025-02-14"},
            {"customer_id": "C003", "name": "M. Kowalski", "country": "PL", "segment": "Consumer", "signup_date": "2025-03-03"},
            {"customer_id": "C004", "name": "J. Müller", "country": "DE", "segment": "Enterprise", "signup_date": "2025-03-18"},
            {"customer_id": "C005", "name": "R. Singh", "country": "IN", "segment": "SMB", "signup_date": "2025-04-25"},
            {"customer_id": "C006", "name": "L. Silva", "country": "BR", "segment": "Consumer", "signup_date": "2025-05-09"},
            {"customer_id": "C007", "name": "N. Johnson", "country": "US", "segment": "Enterprise", "signup_date": "2025-06-21"},
            {"customer_id": "C008", "name": "Y. Chen", "country": "CN", "segment": "Consumer", "signup_date": "2025-07-02"},
        ]
    )

    orders = pd.DataFrame(
        [
            {"order_id": "O100", "customer_id": "C001", "order_date": "2025-12-10", "order_status": "Shipped", "currency": "USD"},
            {"order_id": "O101", "customer_id": "C002", "order_date": "2025-12-11", "order_status": "Delivered", "currency": "SGD"},
            {"order_id": "O102", "customer_id": "C001", "order_date": "2025-12-20", "order_status": "Delivered", "currency": "USD"},
            {"order_id": "O103", "customer_id": "C003", "order_date": "2025-12-22", "order_status": "Shipped", "currency": "EUR"},
            {"order_id": "O104", "customer_id": "C004", "order_date": "2025-12-23", "order_status": "Processing", "currency": "EUR"},
            {"order_id": "O105", "customer_id": "C005", "order_date": "2025-12-24", "order_status": "Cancelled", "currency": "INR"},
            {"order_id": "O106", "customer_id": "C006", "order_date": "2025-12-26", "order_status": "Delivered", "currency": "BRL"},
            {"order_id": "O107", "customer_id": "C002", "order_date": "2025-12-27", "order_status": "Delivered", "currency": "SGD"},
            {"order_id": "O108", "customer_id": "C007", "order_date": "2025-12-28", "order_status": "Shipped", "currency": "USD"},
            {"order_id": "O109", "customer_id": "C008", "order_date": "2025-12-29", "order_status": "Processing", "currency": "CNY"},
            {"order_id": "O110", "customer_id": "C003", "order_date": "2025-12-30", "order_status": "Delivered", "currency": "EUR"},
            {"order_id": "O111", "customer_id": "C001", "order_date": "2025-12-31", "order_status": "Delivered", "currency": "USD"},
        ]
    )

    products = pd.DataFrame(
        [
            {"product_id": "P10", "product": "Keyboard", "category_id": "CAT1", "supplier_id": "SUP1", "list_price": 69.0},
            {"product_id": "P11", "product": "Mouse", "category_id": "CAT1", "supplier_id": "SUP2", "list_price": 39.0},
            {"product_id": "P12", "product": "Monitor", "category_id": "CAT2", "supplier_id": "SUP3", "list_price": 199.0},
            {"product_id": "P13", "product": "USB-C Cable", "category_id": "CAT1", "supplier_id": "SUP1", "list_price": 14.0},
            {"product_id": "P14", "product": "Laptop Stand", "category_id": "CAT3", "supplier_id": "SUP4", "list_price": 49.0},
            {"product_id": "P15", "product": "Webcam", "category_id": "CAT4", "supplier_id": "SUP3", "list_price": 89.0},
            {"product_id": "P16", "product": "Docking Station", "category_id": "CAT3", "supplier_id": "SUP5", "list_price": 149.0},
            {"product_id": "P17", "product": "Desk Lamp", "category_id": "CAT4", "supplier_id": "SUP4", "list_price": 29.0},
            {"product_id": "P18", "product": "External SSD", "category_id": "CAT2", "supplier_id": "SUP5", "list_price": 129.0},
            {"product_id": "P19", "product": "Headset", "category_id": "CAT1", "supplier_id": "SUP2", "list_price": 59.0},
        ]
    )

    order_items = pd.DataFrame(
        [
            {"order_id": "O100", "product_id": "P10", "qty": 1, "unit_price": 65.0},
            {"order_id": "O100", "product_id": "P13", "qty": 2, "unit_price": 12.0},
            {"order_id": "O101", "product_id": "P12", "qty": 1, "unit_price": 189.0},
            {"order_id": "O102", "product_id": "P11", "qty": 1, "unit_price": 35.0},
            {"order_id": "O102", "product_id": "P19", "qty": 1, "unit_price": 55.0},
            {"order_id": "O103", "product_id": "P14", "qty": 1, "unit_price": 45.0},
            {"order_id": "O103", "product_id": "P15", "qty": 1, "unit_price": 82.0},
            {"order_id": "O104", "product_id": "P16", "qty": 2, "unit_price": 139.0},
            {"order_id": "O104", "product_id": "P10", "qty": 1, "unit_price": 67.0},
            {"order_id": "O105", "product_id": "P17", "qty": 1, "unit_price": 26.0},
            {"order_id": "O106", "product_id": "P18", "qty": 1, "unit_price": 119.0},
            {"order_id": "O106", "product_id": "P11", "qty": 1, "unit_price": 38.0},
            {"order_id": "O107", "product_id": "P13", "qty": 3, "unit_price": 13.0},
            {"order_id": "O107", "product_id": "P12", "qty": 1, "unit_price": 195.0},
            {"order_id": "O108", "product_id": "P10", "qty": 1, "unit_price": 68.0},
            {"order_id": "O108", "product_id": "P14", "qty": 1, "unit_price": 47.0},
            {"order_id": "O109", "product_id": "P15", "qty": 1, "unit_price": 86.0},
            {"order_id": "O110", "product_id": "P18", "qty": 2, "unit_price": 125.0},
            {"order_id": "O111", "product_id": "P19", "qty": 1, "unit_price": 57.0},
            {"order_id": "O111", "product_id": "P11", "qty": 2, "unit_price": 36.0},
        ]
    )

    categories = pd.DataFrame(
        [
            {"category_id": "CAT1", "category": "Accessories", "department_id": "DEP1"},
            {"category_id": "CAT2", "category": "Displays & Storage", "department_id": "DEP1"},
            {"category_id": "CAT3", "category": "Workspace", "department_id": "DEP2"},
            {"category_id": "CAT4", "category": "Imaging", "department_id": "DEP2"},
        ]
    )

    departments = pd.DataFrame(
        [
            {"department_id": "DEP1", "department": "Computing"},
            {"department_id": "DEP2", "department": "Office"},
        ]
    )

    suppliers = pd.DataFrame(
        [
            {"supplier_id": "SUP1", "supplier": "Northwind Components", "supplier_country": "US"},
            {"supplier_id": "SUP2", "supplier": "Lion City Peripherals", "supplier_country": "SG"},
            {"supplier_id": "SUP3", "supplier": "Vistula Electronics", "supplier_country": "PL"},
            {"supplier_id": "SUP4", "supplier": "Rhein Workspace", "supplier_country": "DE"},
            {"supplier_id": "SUP5", "supplier": "Pearl River Tech", "supplier_country": "CN"},
        ]
    )

    warehouses = pd.DataFrame(
        [
            {"warehouse_id": "W1", "warehouse_city": "Newark", "warehouse_country": "US"},
            {"warehouse_id": "W2", "warehouse_city": "Singapore", "warehouse_country": "SG"},
            {"warehouse_id": "W3", "warehouse_city": "Frankfurt", "warehouse_country": "DE"},
        ]
    )

    inventory_rows = []
    for wid in ["W1", "W2", "W3"]:
        for pid, base in [
            ("P10", 22),
            ("P11", 35),
            ("P12", 14),
            ("P13", 80),
            ("P14", 18),
            ("P15", 20),
            ("P16", 12),
            ("P17", 40),
            ("P18", 16),
            ("P19", 28),
        ]:
            # Slight deterministic variation per warehouse to keep it "realistic"
            bump = 0 if wid == "W1" else (3 if wid == "W2" else -2)
            inventory_rows.append(
                {
                    "warehouse_id": wid,
                    "product_id": pid,
                    "on_hand_qty": max(0, base + bump),
                    "reorder_point": 10,
                }
            )
    inventory = pd.DataFrame(inventory_rows)

    carriers = pd.DataFrame(
        [
            {"carrier_id": "CR1", "carrier": "DHL"},
            {"carrier_id": "CR2", "carrier": "UPS"},
            {"carrier_id": "CR3", "carrier": "FedEx"},
        ]
    )

    shipments = pd.DataFrame(
        [
            {"shipment_id": "S900", "order_id": "O100", "warehouse_id": "W1", "carrier_id": "CR1", "ship_date": "2025-12-11"},
            {"shipment_id": "S901", "order_id": "O101", "warehouse_id": "W2", "carrier_id": "CR2", "ship_date": "2025-12-12"},
            {"shipment_id": "S902", "order_id": "O102", "warehouse_id": "W1", "carrier_id": "CR3", "ship_date": "2025-12-21"},
            {"shipment_id": "S903", "order_id": "O103", "warehouse_id": "W3", "carrier_id": "CR1", "ship_date": "2025-12-23"},
            {"shipment_id": "S904", "order_id": "O104", "warehouse_id": "W3", "carrier_id": "CR2", "ship_date": "2025-12-24"},
            {"shipment_id": "S905", "order_id": "O105", "warehouse_id": "W2", "carrier_id": "CR3", "ship_date": "2025-12-24"},
            {"shipment_id": "S906", "order_id": "O106", "warehouse_id": "W1", "carrier_id": "CR2", "ship_date": "2025-12-27"},
            {"shipment_id": "S907", "order_id": "O107", "warehouse_id": "W2", "carrier_id": "CR1", "ship_date": "2025-12-28"},
            {"shipment_id": "S908", "order_id": "O108", "warehouse_id": "W1", "carrier_id": "CR3", "ship_date": "2025-12-29"},
            {"shipment_id": "S909", "order_id": "O109", "warehouse_id": "W2", "carrier_id": "CR2", "ship_date": "2025-12-30"},
            {"shipment_id": "S910", "order_id": "O110", "warehouse_id": "W3", "carrier_id": "CR1", "ship_date": "2025-12-31"},
            {"shipment_id": "S911", "order_id": "O111", "warehouse_id": "W1", "carrier_id": "CR2", "ship_date": "2026-01-02"},
        ]
    )

    payment_methods = pd.DataFrame(
        [
            {"payment_method_id": "PM1", "payment_method": "Credit Card"},
            {"payment_method_id": "PM2", "payment_method": "PayPal"},
            {"payment_method_id": "PM3", "payment_method": "Bank Transfer"},
        ]
    )

    payments = pd.DataFrame(
        [
            {"payment_id": "PAY900", "order_id": "O100", "payment_method_id": "PM1", "amount": 89.0, "payment_date": "2025-12-10"},
            {"payment_id": "PAY901", "order_id": "O101", "payment_method_id": "PM2", "amount": 189.0, "payment_date": "2025-12-11"},
            {"payment_id": "PAY902", "order_id": "O102", "payment_method_id": "PM1", "amount": 90.0, "payment_date": "2025-12-20"},
            {"payment_id": "PAY903", "order_id": "O103", "payment_method_id": "PM3", "amount": 127.0, "payment_date": "2025-12-22"},
            {"payment_id": "PAY904", "order_id": "O104", "payment_method_id": "PM1", "amount": 345.0, "payment_date": "2025-12-23"},
            {"payment_id": "PAY905", "order_id": "O105", "payment_method_id": "PM2", "amount": 26.0, "payment_date": "2025-12-24"},
            {"payment_id": "PAY906", "order_id": "O106", "payment_method_id": "PM1", "amount": 157.0, "payment_date": "2025-12-26"},
            {"payment_id": "PAY907", "order_id": "O107", "payment_method_id": "PM2", "amount": 234.0, "payment_date": "2025-12-27"},
            {"payment_id": "PAY908", "order_id": "O108", "payment_method_id": "PM3", "amount": 115.0, "payment_date": "2025-12-28"},
            {"payment_id": "PAY909", "order_id": "O109", "payment_method_id": "PM1", "amount": 86.0, "payment_date": "2025-12-29"},
            {"payment_id": "PAY910", "order_id": "O110", "payment_method_id": "PM2", "amount": 250.0, "payment_date": "2025-12-30"},
            {"payment_id": "PAY911", "order_id": "O111", "payment_method_id": "PM1", "amount": 129.0, "payment_date": "2025-12-31"},
        ]
    )

    returns = pd.DataFrame(
        [
            {"return_id": "R100", "order_id": "O102", "product_id": "P19", "return_date": "2026-01-03", "return_reason": "Defective", "return_amount": 55.0},
            {"return_id": "R101", "order_id": "O103", "product_id": "P15", "return_date": "2026-01-05", "return_reason": "Changed mind", "return_amount": 82.0},
            {"return_id": "R102", "order_id": "O104", "product_id": "P16", "return_date": "2026-01-06", "return_reason": "Not compatible", "return_amount": 139.0},
            {"return_id": "R103", "order_id": "O106", "product_id": "P11", "return_date": "2026-01-07", "return_reason": "Late delivery", "return_amount": 38.0},
            {"return_id": "R104", "order_id": "O107", "product_id": "P12", "return_date": "2026-01-08", "return_reason": "Damaged box", "return_amount": 195.0},
            {"return_id": "R105", "order_id": "O111", "product_id": "P11", "return_date": "2026-01-09", "return_reason": "Wrong item", "return_amount": 36.0},
        ]
    )

    reviews = pd.DataFrame(
        [
            {"review_id": "RV001", "customer_id": "C001", "product_id": "P10", "rating": 5, "review_date": "2025-12-18"},
            {"review_id": "RV002", "customer_id": "C002", "product_id": "P12", "rating": 4, "review_date": "2025-12-20"},
            {"review_id": "RV003", "customer_id": "C003", "product_id": "P14", "rating": 5, "review_date": "2025-12-26"},
            {"review_id": "RV004", "customer_id": "C004", "product_id": "P16", "rating": 4, "review_date": "2025-12-28"},
            {"review_id": "RV005", "customer_id": "C005", "product_id": "P17", "rating": 3, "review_date": "2025-12-29"},
            {"review_id": "RV006", "customer_id": "C006", "product_id": "P18", "rating": 5, "review_date": "2025-12-30"},
            {"review_id": "RV007", "customer_id": "C002", "product_id": "P13", "rating": 4, "review_date": "2025-12-31"},
            {"review_id": "RV008", "customer_id": "C007", "product_id": "P19", "rating": 4, "review_date": "2026-01-01"},
            {"review_id": "RV009", "customer_id": "C008", "product_id": "P15", "rating": 5, "review_date": "2026-01-02"},
            {"review_id": "RV010", "customer_id": "C001", "product_id": "P11", "rating": 4, "review_date": "2026-01-03"},
        ]
    )

    return {
        "customers": customers,
        "orders": orders,
        "order_items": order_items,
        "products": products,
        "categories": categories,
        "departments": departments,
        "suppliers": suppliers,
        "warehouses": warehouses,
        "inventory": inventory,
        "shipments": shipments,
        "carriers": carriers,
        "payments": payments,
        "payment_methods": payment_methods,
        "returns": returns,
        "reviews": reviews,
    }


@st.cache_data(show_spinner=False)
def bundled_query() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"order_id": "O100", "customer_id": "C001", "country": "US"},
            {"order_id": "O101", "customer_id": "C002", "country": "SG"},
            {"order_id": "O103", "customer_id": "C003", "country": "PL"},
            {"order_id": "O104", "customer_id": "C004", "country": "DE"},
            {"order_id": "O108", "customer_id": "C007", "country": "US"},
        ]
    )


def read_csv_bytes(b: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(b))


@st.cache_data(show_spinner=False)
def load_datalake_from_zip(zip_bytes: bytes) -> Dict[str, pd.DataFrame]:
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    tables: Dict[str, pd.DataFrame] = {}
    for info in zf.infolist():
        if info.is_dir():
            continue
        if not info.filename.lower().endswith(".csv"):
            continue
        name = info.filename.split("/")[-1].rsplit(".", 1)[0].strip()
        if not name:
            continue
        df = read_csv_bytes(zf.read(info))
        if df.empty or df.shape[1] == 0:
            continue
        tables[name] = df
    return tables


def token_for_uploaded_files(files: Sequence) -> Optional[str]:
    if not files:
        return None
    h = hashlib.md5()
    for f in files:
        name = str(getattr(f, "name", "")).encode("utf-8", errors="ignore")
        h.update(name)
        h.update(b"\0")
    return h.hexdigest()


def ensure_nonempty_tables(tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    cleaned = {}
    for k, df in tables.items():
        if df is None or df.empty or df.shape[1] == 0:
            continue
        df2 = df.copy()
        # Drop fully blank rows (common in CSV exports) so they don't render or affect joins.
        obj_cols = df2.select_dtypes(include=["object"]).columns
        if len(obj_cols) > 0:
            df2[obj_cols] = df2[obj_cols].astype("string").apply(lambda s: s.str.strip())
            df2[obj_cols] = df2[obj_cols].replace(r"^\s*$", pd.NA, regex=True)
        df2 = df2.dropna(how="all")
        if df2.empty or df2.shape[1] == 0:
            continue
        cleaned[k] = df2
    return cleaned


def drop_fully_blank_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or df.shape[1] == 0:
        return df
    df2 = df.copy()
    # Normalize whitespace-only strings to NA for object/string columns, then drop rows that
    # are blank across all columns.
    obj_cols = df2.select_dtypes(include=["object", "string"]).columns
    if len(obj_cols) > 0:
        df2[obj_cols] = df2[obj_cols].astype("string").apply(lambda s: s.str.strip())
        df2[obj_cols] = df2[obj_cols].replace(r"^\s*$", pd.NA, regex=True)
    df2 = df2.dropna(how="all")
    return df2


def format_path(path: Sequence[str], edges: Sequence[EdgeInfo]) -> str:
    parts = [path[0]]
    for e, nxt in zip(edges, path[1:]):
        parts.append(f"--[{e.key}, {e.score:.2f}]-- {nxt}")
    return " ".join(parts)


def path_to_dot(nodes: Sequence[str], edges: Sequence[EdgeInfo], tables: Optional[Dict[str, pd.DataFrame]] = None) -> str:
    lines = [
        "graph P {",
        # Tighter layout: reduce spacing between ranks/nodes for shorter edges.
        '  graph [rankdir="LR", bgcolor="white", ranksep="0.14", nodesep="0.10"];',
        # More compact nodes so long paths don't waste vertical space.
        '  node [shape="box", style="rounded,filled", fillcolor="#f7f7f7", color="#444444", fontname="Helvetica", fontsize="7", margin="0.03,0.015", height="0.12", width="0.26"];',
        # Smaller edge label font (about half default) for compact rendering.
        '  edge [color="#d62728", fontname="Helvetica", fontsize="7", penwidth="1.4"];',
    ]
    for n in nodes:
        lines.append(f'  "{n}";')
    for i, (e, nxt) in enumerate(zip(edges, nodes[1:])):
        # Joins are non-directional: connect current node to next node
        # Get the join key column name from each table
        # Currently e.key is the common column name, but we show it for both tables
        left_join_key = e.key
        right_join_key = e.key
        
        # Show just the two join keys separated by comma
        label = f"{left_join_key}, {right_join_key}"
        
        # Use undirected edge (--) instead of directed (->)
        current_node = nodes[i]
        lines.append(f'  "{current_node}" -- "{nxt}" [label="{label}"];')
    lines.append("}")
    return "\n".join(lines)


def path_to_dot_highlighted(
    nodes: Sequence[str],
    edges: Sequence[EdgeInfo],
    tables: Optional[Dict[str, pd.DataFrame]] = None,
    highlight_blue: bool = False,
    highlight_green_nodes: Optional[Set[str]] = None,
) -> str:
    """Generate DOT source for a path with optional highlighting.
    
    Args:
        nodes: List of table names in the path
        edges: List of edges connecting the nodes
        tables: Optional dict of table DataFrames
        highlight_blue: If True, highlight the entire path in blue (row match)
        highlight_green_nodes: Set of node names to highlight in green (column match)
    
    Returns:
        DOT source string
    """
    highlight_green_nodes = highlight_green_nodes or set()
    
    # Base colors
    default_fill = "#f7f7f7"
    default_border = "#444444"
    default_edge = "#d62728"
    
    # Highlight colors
    blue_fill = "#DBEAFE"  # Light blue background
    blue_border = "#2563EB"  # Blue border
    blue_edge = "#2563EB"  # Blue edges
    green_fill = "#D1FAE5"  # Light green background
    green_border = "#059669"  # Green border
    
    edge_color = blue_edge if highlight_blue else default_edge
    
    lines = [
        "graph P {",
        # Tighter layout: reduce spacing between ranks/nodes for shorter edges.
        '  graph [rankdir="LR", bgcolor="white", ranksep="0.14", nodesep="0.10"];',
        # Smaller edge label font (about half default) for compact rendering.
        f'  edge [color="{edge_color}", fontname="Helvetica", fontsize="7", penwidth="1.4"];',
    ]
    
    for n in nodes:
        # Determine node styling
        if n in highlight_green_nodes:
            fill = green_fill
            border = green_border
        elif highlight_blue:
            fill = blue_fill
            border = blue_border
        else:
            fill = default_fill
            border = default_border
        
        lines.append(
            f'  "{n}" [shape="box", style="rounded,filled", fillcolor="{fill}", color="{border}", '
            f'fontname="Helvetica", fontsize="7", margin="0.03,0.015", height="0.12", width="0.26"];'
        )
    
    for i, (e, nxt) in enumerate(zip(edges, nodes[1:])):
        left_join_key = e.key
        right_join_key = e.key
        label = f"{left_join_key}, {right_join_key}"
        current_node = nodes[i]
        lines.append(f'  "{current_node}" -- "{nxt}" [label="{label}"];')
    
    lines.append("}")
    return "\n".join(lines)


def render_graphviz_fixed_height(dot: str, *, height_px: int = 220) -> None:
    """Render a Graphviz DOT string at a fixed height with horizontal scroll.

    Streamlit's native `st.graphviz_chart(..., use_container_width=True)` scales graphs
    to the container width, which can shrink nodes when paths get long. This renderer
    keeps a fixed height and allows the graph to grow left-to-right.
    """
    import streamlit.components.v1 as components

    # Stable DOM id for this DOT string (avoid collisions inside an iframe).
    dom_id = f"sd-gv-{token_for_bytes(dot.encode('utf-8', errors='ignore'))[:12]}"
    dot_json = json.dumps(dot)

    # Fit inside a fixed-height viewport, allow horizontal scrolling.
    # Note: this runs in an iframe, so the CSS/JS is isolated per render.
    inner_h = max(80, int(height_px) - 10)
    component_h = int(height_px) + 28

    html = f"""
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
    <script src="https://cdn.jsdelivr.net/npm/viz.js@2.1.2/viz.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/viz.js@2.1.2/full.render.js"></script>
    <style>
      html, body {{
        margin: 0;
        padding: 0;
        background: transparent;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      }}
      .sd-gv-wrap {{
        width: 100%;
        height: {int(height_px)}px;
        overflow-x: auto;
        overflow-y: hidden;
        background: rgba(255, 255, 255, 0.65);
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 8px;
      }}
      /* Override SVG attributes so it doesn't shrink to fit width */
      .sd-gv-wrap svg {{
        height: {inner_h}px !important;
        width: auto !important;
        max-width: none !important;
        display: block;
      }}
    </style>
  </head>
  <body>
    <div class="sd-gv-wrap">
      <div id="{dom_id}"></div>
    </div>
    <script>
      (function() {{
        const dot = {dot_json};
        const viz = new Viz();
        const mount = document.getElementById("{dom_id}");
        viz.renderSVGElement(dot)
          .then(function(el) {{
            el.style.height = "{inner_h}px";
            el.style.width = "auto";
            el.style.maxWidth = "none";
            mount.innerHTML = "";
            mount.appendChild(el);
          }})
          .catch(function(err) {{
            mount.innerHTML = '<div style="color:#B91C1C; font-size:13px;">Error rendering join path diagram</div>';
            console.error(err);
          }});
      }})();
    </script>
  </body>
</html>
""".strip()

    components.html(html, height=component_h, scrolling=False)


def token_for_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def simulate_preprocessing(
    steps: Sequence[str],
    seconds_per_step: float = 0.25,
    *,
    container=None,
    log_key: str = "preprocess_log",
) -> None:
    st.session_state[log_key] = []

    scope = container.container() if container is not None else nullcontext()
    with scope:
        progress = st.progress(0, text="Preprocessing...")
        with st.status("Preprocessing", expanded=True) as status:
            for i, step in enumerate(steps, start=1):
                with st.spinner(step):
                    time.sleep(seconds_per_step)
                line = f"✓ {step}"
                st.session_state[log_key].append(line)
                status.write(line)
                progress.progress(int(100 * i / max(len(steps), 1)), text="Preprocessing...")
            status.update(label="Preprocessing complete", state="complete", expanded=False)
        progress.empty()


def render_brand_header() -> None:
    # Simple inline SVG mark for SemDisc (no external assets required).
    svg = """
<svg width="44" height="44" viewBox="0 0 44 44" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="SemDisc logo">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#4F46E5"/>
      <stop offset="1" stop-color="#06B6D4"/>
    </linearGradient>
  </defs>
  <rect x="2" y="2" width="40" height="40" rx="10" fill="url(#g)"/>
  <circle cx="20" cy="20" r="9" fill="none" stroke="white" stroke-width="3.2"/>
  <line x1="26.5" y1="26.5" x2="33" y2="33" stroke="white" stroke-width="3.2" stroke-linecap="round"/>
  <path d="M16 19c2-3 8-3 10 0" fill="none" stroke="white" stroke-width="2.6" stroke-linecap="round"/>
</svg>
""".strip()

    c1, c2 = st.columns([0.12, 0.88], vertical_alignment="center")
    with c1:
        st.markdown(svg, unsafe_allow_html=True)
    with c2:
        st.markdown(
            "<div style='font-size:24px; font-weight:750; line-height:1.05;'>SemDisc</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='font-size:0.85rem; color:#6B7280; margin-top:2px;'>"
            "Semantic discovery of joinable tables and join paths."
            "</div>",
            unsafe_allow_html=True,
        )


def render_stage_stepper(current_idx: int, stage_labels: Sequence[str]) -> None:
    # Render an in-page (main area) navigation progress indicator.
    # This replaces the older sidebar-based vertical stepper.
    n = max(1, len(stage_labels))
    idx = int(max(0, min(n - 1, current_idx)))
    pct = int(round(100.0 * float(idx + 1) / float(n)))

    css = """
<style>
  .sd-progress-wrap { margin: 0.75rem 0 1.0rem 0; }
  .sd-bar { height: 10px; border-radius: 999px; background: #E5E7EB; overflow: hidden; margin-top: 0px; }
  .sd-bar > div { height: 100%; width: 0%; background: linear-gradient(90deg, #F97316 0%, #C4B5FD 60%, #F97316 100%); border-radius: 999px; }
  .sd-hstepper { display: grid; grid-template-columns: repeat(var(--sd-n), 1fr); gap: 10px; margin-top: 12px; }
  .sd-hstep { display:flex; flex-direction:column; align-items:center; gap: 6px; text-align:center; }
  .sd-dot { width: 14px; height: 14px; border-radius: 999px; border: 2px solid #9CA3AF; background: #ffffff; box-sizing: border-box; }
  .sd-dot.done { background: #F97316; border-color: #EA580C; }
  .sd-dot.current { background: #F97316; border-color: #EA580C; box-shadow: 0 0 0 3px rgba(249, 115, 22, 0.20); }
  .sd-dot.todo { background: #F5F3FF; border-color: #C4B5FD; }
  .sd-step-label { font-size: 0.86rem; line-height: 1.15; color: #4B5563; }
  .sd-step-label.current { color: #111827; font-weight: 750; }
  .sd-step-label.done { color: #111827; }
</style>
""".strip()

    parts = [f'<div class="sd-progress-wrap" style="--sd-n:{n}">']
    parts.append(f'<div class="sd-bar"><div style="width:{pct}%;"></div></div>')

    parts.append('<div class="sd-hstepper">')
    for i, label in enumerate(stage_labels):
        if i < idx:
            dot_cls = "done"
            lab_cls = "done"
        elif i == idx:
            dot_cls = "current"
            lab_cls = "current"
        else:
            dot_cls = "todo"
            lab_cls = ""
        parts.append('<div class="sd-hstep">')
        parts.append(f'<div class="sd-dot {dot_cls}"></div>')
        parts.append(f'<div class="sd-step-label {lab_cls}">{label}</div>')
        parts.append("</div>")
    parts.append("</div>")
    parts.append("</div>")

    st.markdown(css + "\n" + "\n".join(parts), unsafe_allow_html=True)


def goto_stage(idx: int) -> None:
    st.session_state.page_idx = int(max(0, min(3, idx)))
    st.rerun()

