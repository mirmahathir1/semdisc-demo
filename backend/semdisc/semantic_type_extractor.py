"""Semantic-type extraction for a data lake's columns.

A *semantic type* is an elaborate, human-readable name for a column (e.g. the
``customer_id`` column of the ``orders`` table has semantic type ``"Customer
Identifier"``). These power semantic joins in :mod:`semdisc.join_graph`.

The workflow is cache-first:

* :func:`load_or_extract_semantic_types` looks for a ``semantic_types.json``
  cache (one per data lake / dataset). If present, it is loaded and returned
  without any LLM call.
* If the cache is missing, the LLM is asked to name every column of every
  table, the result is persisted to the cache, and then returned.

The on-disk format is a flat list so it is easy to read and hand-edit::

    [
      {"table": "customers", "column": "customer_id", "semantic_type": "Customer Identifier"},
      ...
    ]

In memory we expose it as ``{(table, column): semantic_type}`` to match how
:data:`semdisc.join_graph.SEMANTIC_TYPES` is consumed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

SemanticTypeMap = Dict[Tuple[str, str], str]

# Filename for the per-data-lake cache.
SEMANTIC_TYPES_FILENAME = "semantic_types.json"

# Sampling controls for the column-value preview shown to the LLM. Mirrors the
# reference extractor: a handful of frequent values, truncated, per column.
_SAMPLE_COUNT = 20
_SAMPLE_CHAR_LIMIT = 150

_OPENAI_MODEL = os.getenv("SEMDISC_OPENAI_MODEL", "gpt-4o")

_PROMPT_TEMPLATE = """\
Below is a table name followed by a list of column ids and a sample of their \
values separated by commas. Give a detailed, elaborate, human-readable \
semantic type name for each column. Return ONLY a JSON array in this exact \
format, with no prose or code fences:
[{{"column": "<column id>", "semantic_type": "<semantic type name>"}}, ...]

input:

{input}
"""


# --------------------------------------------------------------------------- #
# Cache (JSON) I/O
# --------------------------------------------------------------------------- #
def _flat_to_map(records: list[dict]) -> SemanticTypeMap:
    result: SemanticTypeMap = {}
    for record in records:
        table = str(record["table"])
        column = str(record["column"])
        result[(table, column)] = str(record["semantic_type"])
    return result


def _map_to_flat(semantic_types: SemanticTypeMap) -> list[dict]:
    return [
        {"table": table, "column": column, "semantic_type": semantic_type}
        for (table, column), semantic_type in semantic_types.items()
    ]


def load_semantic_types(cache_path: Path) -> SemanticTypeMap | None:
    """Return the cached semantic types, or ``None`` if the cache is absent."""
    if not cache_path.exists():
        return None
    with cache_path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    return _flat_to_map(records)


def save_semantic_types(cache_path: Path, semantic_types: SemanticTypeMap) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(_map_to_flat(semantic_types), handle, indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# LLM extraction
# --------------------------------------------------------------------------- #
def _column_value_sample(df: pd.DataFrame, column: str) -> str:
    values = (
        df[column]
        .dropna()
        .astype(str)
        .value_counts()
        .head(_SAMPLE_COUNT)
        .index.tolist()
    )
    joined = ", ".join(values)[:_SAMPLE_CHAR_LIMIT].replace("\n", " ")
    return f"id: {column}\nsample values: {joined}"


def _build_table_prompt(table: str, df: pd.DataFrame) -> str:
    columns = "\n\n".join(_column_value_sample(df, str(col)) for col in df.columns)
    return _PROMPT_TEMPLATE.format(input=f"table name: {table}\n\n{columns}")


def _parse_llm_json(raw: str) -> list[dict]:
    """Extract the JSON array from an LLM response, tolerating code fences."""
    text = raw.strip()
    if text.startswith("```"):
        # Drop the first fence line and any trailing fence.
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON array found in LLM response: {raw!r}")
    return json.loads(text[start : end + 1])


def _extract_table_semantic_types(client, table: str, df: pd.DataFrame) -> Dict[str, str]:
    """Call the LLM once for a single table; return ``{column: semantic_type}``."""
    prompt = _build_table_prompt(table, df)
    response = client.chat.completions.create(
        model=_OPENAI_MODEL,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    records = _parse_llm_json(response.choices[0].message.content)
    parsed = {str(record["column"]): str(record["semantic_type"]) for record in records}

    # The LLM may rename/miss columns; guarantee one entry per real column by
    # falling back to a title-cased version of the column id.
    result: Dict[str, str] = {}
    for col in map(str, df.columns):
        result[col] = parsed.get(col) or col.replace("_", " ").title()
    return result


def extract_semantic_types(tables: Dict[str, pd.DataFrame]) -> SemanticTypeMap:
    """Run the LLM over every table/column. Requires ``OPENAI_API_KEY``."""
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise RuntimeError(
            "openai package is required for semantic-type extraction. "
            "Install it with `pip install openai`."
        ) from exc

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set; cannot extract semantic types via the LLM."
        )

    client = OpenAI()
    semantic_types: SemanticTypeMap = {}
    for table, df in tables.items():
        for column, semantic_type in _extract_table_semantic_types(client, table, df).items():
            semantic_types[(table, column)] = semantic_type
    return semantic_types


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def load_or_extract_semantic_types(
    tables: Dict[str, pd.DataFrame],
    cache_path: Path,
    *,
    force: bool = False,
) -> SemanticTypeMap:
    """Return semantic types for ``tables``, using the cache when possible.

    If ``cache_path`` exists (and ``force`` is False) the cached JSON is loaded
    and returned. Otherwise the LLM is invoked, the result is written to
    ``cache_path``, and then returned.
    """
    if not force:
        cached = load_semantic_types(cache_path)
        if cached is not None:
            return cached

    semantic_types = extract_semantic_types(tables)
    save_semantic_types(cache_path, semantic_types)
    return semantic_types
