from __future__ import annotations

import os
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


def render_query_table_selector(
    *,
    df: pd.DataFrame,
    selected_row: Optional[int],
    selected_column: Optional[str],
    height_px: int = 260,
    title: str = "Query table",
    key: str = "query_table_selector",
) -> Optional[Dict[str, Any]]:
    """
    Render an interactive query table that supports:
    - single-row selection (click a row)
    - single-column selection (click a column header)

    Returns a dict:
      { "selected_row": int | null, "selected_column": str | null }
    """

    component_dir = os.path.join(os.path.dirname(__file__), "query_table_component")
    query_table = components.declare_component("query_table_selector", path=component_dir)

    # Render values as strings for display stability (avoid JSON serialization edge cases).
    safe_df = df.copy()
    safe_df = safe_df.where(pd.notna(safe_df), "")
    safe_df = safe_df.astype(str)

    payload = {
        "columns": [str(c) for c in safe_df.columns.tolist()],
        "rows": safe_df.values.tolist(),
    }

    picked: Any = query_table(
        data=payload,
        title=str(title),
        height_px=int(height_px),
        selected_row=selected_row,
        selected_column=selected_column,
        height=int(height_px),
        key=key,
        default=None,
    )

    if isinstance(picked, dict):
        return picked
    return None

