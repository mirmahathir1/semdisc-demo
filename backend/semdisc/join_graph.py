from __future__ import annotations

import itertools
import random
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .models import EdgeInfo
from .semantic_minhash import compute_column_minhashes, semantic_type_cosine
from .semantic_minhash import jaccard as minhash_jaccard


SEMANTIC_TYPES: Dict[Tuple[str, str], str] = {
    # orders data lake
    ("customers", "customer_id"): "Customer Identifier",
    ("customers", "name"): "Customer Full Name",
    ("customers", "country"): "Customer Country",
    ("customers", "customer_no"): "Customer Number",
    ("orders", "order_id"): "Order Identifier",
    ("orders", "customer_id"): "Customer Identifier",
    ("orders", "order_date"): "Order Date",
    ("orders", "ship_country"): "Customer Country",
    ("orders", "preferred_carrier"): "Carrier Name",
    ("orders", "customer_no"): "Customer Number",
    ("orders", "order_no"): "Order Number",
    ("order_items", "order_id"): "Order Identifier",
    ("order_items", "product_id"): "Product Identifier",
    ("order_items", "qty"): "Quantity Ordered",
    ("order_items", "item_label"): "Product Name",
    ("order_items", "product_no"): "Product Number",
    ("order_items", "order_no"): "Order Number",
    ("products", "product_id"): "Product Identifier",
    ("products", "product"): "Product Name",
    ("products", "category_id"): "Product Category Identifier",
    ("products", "product_family"): "Product Category Name",
    ("products", "product_no"): "Product Number",
    ("products", "category_no"): "Product Category Number",
    ("categories", "category_id"): "Product Category Identifier",
    ("categories", "category"): "Product Category Name",
    ("categories", "category_no"): "Product Category Number",
    ("shipments", "order_id"): "Order Identifier",
    ("shipments", "shipment_id"): "Shipment Identifier",
    ("shipments", "carrier"): "Carrier Name",
    ("shipments", "order_no"): "Order Number",
    # country data lake
    ("GDP_Per_Capita_Completed", "country"): "Country Name",
    ("GDP_Per_Capita_Completed", "gdp (in trillions USD)"): "Country GDP (Trillions USD)",
    ("GDP_Per_Capita_Completed", "region"): "Continent Name",
    ("GDP_Per_Capita_Completed", "official_language"): "Official Language",
    ("GDP_Per_Capita_Completed", "internet_penetration"): "Internet Penetration Percent",
    ("GDP_Per_Capita_Completed", "currency"): "Currency Code",
    ("GDP_Per_Capita_Completed", "population_millions"): "Population (Millions)",
    ("GDP_Per_Capita_Completed", "gdp_per_capita_usd"): "GDP Per Capita USD",
    ("census_data", "country"): "Country Name",
    ("census_data", "pop_data"): "Population (Millions)",
    ("census_data", "data_year"): "Data Year",
    ("census_data", "source"): "Data Source",
    ("climate_metrics", "country"): "Country Name",
    ("climate_metrics", "temperature"): "Average Temperature",
    ("continent_gdp", "Continent"): "Continent Name",
    ("continent_gdp", "GDP (in Trillions USD)"): "Continent GDP (Trillions USD)",
    ("continent_mapping", "country"): "Country Name",
    ("continent_mapping", "continent"): "Continent Name",
    ("country_currency_strength", "country"): "Country Name",
    ("country_currency_strength", "currency"): "Currency Code",
    ("country_currency_strength", "currency_strength"): "Currency Strength",
    ("currency", "country"): "Country Name",
    ("currency", "currency"): "Currency Code",
    ("economic_metrics", "country"): "Country Name",
    ("economic_metrics", "inflation_rate_annual_percent"): "Inflation Rate Percent",
    ("gini_index", "country"): "Country Name",
    ("gini_index", "gini_coefficient"): "Gini Coefficient",
    ("global_tourism", "country"): "Country Name",
    ("global_tourism", "tourism"): "Tourist Arrivals",
    ("health_indicators", "country"): "Country Name",
    ("health_indicators", "life_expectancy"): "Life Expectancy",
    ("health_indicators", "healthcare_expenditures"): "Healthcare Expenditure",
    ("literacy_rates", "country"): "Country Name",
    ("literacy_rates", "literacy_rate"): "Literacy Rate Percent",
    ("literacy_rates", "data_source"): "Data Source",
    ("renewable_energy", "country"): "Country Name",
    ("renewable_energy", "renewable_energy_percent"): "Renewable Energy Percent",
    ("renewable_energy", "data_year"): "Data Year",
    ("renewable_energy", "source"): "Data Source",
    ("tech_market_cap", "company"): "Company Name",
    ("tech_market_cap", "market_cap_billion_usd"): "Market Cap (Billion USD)",
    ("tech_market_cap", "headquarters"): "Headquarters Country",
    ("tech_market_cap", "as_of_year"): "Data Year",
    ("university_acceptance", "university"): "University Name",
    ("university_acceptance", "acceptance_rate_percent"): "Acceptance Rate Percent",
    ("university_acceptance", "country"): "Country Name",
    ("university_acceptance", "latest_data_year"): "Data Year",
}


QUERY_SEMANTIC_COLUMN_MAP = {
    # orders data lake
    "Order Identifier": "order_id",
    "Customer IDs": "customer_id",
    "Customer Identifier": "customer_id",
    "Country Name": "country",
    # country data lake
    "Currency Code": "currency",
    "Continent Name": "continent",
}


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


def _append_edge_pairs_by_score(
    graph: Dict[str, List[EdgeInfo]],
    edge_pairs: List[Tuple[EdgeInfo, EdgeInfo]],
) -> None:
    by_score: Dict[float, List[Tuple[EdgeInfo, EdgeInfo]]] = defaultdict(list)
    for forward, reverse in edge_pairs:
        by_score[forward.score].append((forward, reverse))

    for score in sorted(by_score, reverse=True):
        tied = by_score[score]
        if len(tied) > 1:
            random.shuffle(tied)
        for forward, reverse in tied:
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

        _append_edge_pairs_by_score(graph, edge_pairs)

    return graph
