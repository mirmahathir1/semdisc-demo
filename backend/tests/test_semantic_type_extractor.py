from __future__ import annotations

import json

import pandas as pd
import pytest

from semdisc.semantic_type_extractor import (
    load_or_extract_semantic_types,
    load_semantic_types,
    save_semantic_types,
)


def test_save_then_load_roundtrips_the_map(tmp_path):
    cache = tmp_path / "semantic_types.json"
    semantic_types = {
        ("orders", "order_id"): "Order Identifier",
        ("orders", "customer_id"): "Customer Identifier",
    }

    save_semantic_types(cache, semantic_types)
    assert load_semantic_types(cache) == semantic_types

    # On-disk format is a flat, hand-editable list of records.
    records = json.loads(cache.read_text())
    assert {"table", "column", "semantic_type"} == set(records[0])


def test_load_returns_none_when_cache_absent(tmp_path):
    assert load_semantic_types(tmp_path / "missing.json") is None


def test_load_or_extract_uses_cache_without_calling_llm(tmp_path):
    cache = tmp_path / "semantic_types.json"
    cached = {("t", "a"): "Alpha"}
    save_semantic_types(cache, cached)

    tables = {"t": pd.DataFrame({"a": [1, 2]})}
    # No OPENAI_API_KEY / openai needed: a cache hit must short-circuit the LLM.
    assert load_or_extract_semantic_types(tables, cache) == cached


def test_load_or_extract_raises_when_no_key_and_no_cache(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    tables = {"t": pd.DataFrame({"a": [1, 2]})}
    with pytest.raises(RuntimeError):
        load_or_extract_semantic_types(tables, tmp_path / "missing.json")
