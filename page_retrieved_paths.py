from __future__ import annotations

import html
import json
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import os
from typing import Dict, List, Set, Tuple

from components.heatmap import HeatmapCell, render_joinpath_heatmap

from utils import (
    EdgeInfo,
    attach_query_node,
    build_join_graph,
    format_path,
    goto_stage,
    materialize_path,
    path_to_dot,
    path_to_dot_highlighted,
    render_graphviz_fixed_height,
    simple_paths_topk,
    SEMANTIC_TYPES,
)


def get_tables_with_column_value(
    tables: Dict[str, pd.DataFrame],
    column_name: str,
    query_df: pd.DataFrame,
    selected_row: int,
) -> Set[str]:
    """Find tables that contain a column matching the query column's value.
    
    Returns set of table names where the column exists and contains
    the value from the selected row.
    """
    matching_tables = set()
    
    if selected_row is None or selected_row < 0 or selected_row >= len(query_df):
        return matching_tables
    
    if column_name not in query_df.columns:
        return matching_tables
    
    query_value = str(query_df.iloc[selected_row][column_name]).strip()
    if not query_value or query_value.lower() == 'nan':
        return matching_tables
    
    # Check each table for matching column and value
    for table_name, df in tables.items():
        if column_name in df.columns:
            # Check if any row contains the query value
            col_values = df[column_name].astype(str).str.strip()
            if query_value in col_values.values:
                matching_tables.add(table_name)
    
    return matching_tables


def check_row_match(
    path_nodes: List[str],
    path_edges: List[EdgeInfo],
    tables: Dict[str, pd.DataFrame],
    query_df: pd.DataFrame,
    selected_row: int,
) -> bool:
    """Check if the path contains all values from the selected row.
    
    Returns True if all non-null query values are found in the materialized path.
    """
    if selected_row is None or selected_row < 0 or selected_row >= len(query_df):
        return False
    
    try:
        materialized = materialize_path(tables, path_nodes, path_edges)
    except Exception:
        return False
    
    if materialized.empty:
        return False
    
    # Get query row values
    query_row = query_df.iloc[selected_row]
    
    # Check each query column value
    for col in query_df.columns:
        val = str(query_row[col]).strip()
        if not val or val.lower() == 'nan':
            continue
        
        # Check if this value exists somewhere in the materialized table
        found = False
        for mat_col in materialized.columns:
            if val in materialized[mat_col].astype(str).str.strip().values:
                found = True
                break
        
        if not found:
            return False
    
    return True


def render_join_path_card(
    rank: int,
    nodes: List[str],
    edges: List[EdgeInfo],
    tables: Dict[str, pd.DataFrame],
    highlight_blue: bool = False,
    highlight_green_nodes: Set[str] = None,
) -> str:
    """Generate HTML for a join path card with Graphviz visualization."""
    highlight_green_nodes = highlight_green_nodes or set()
    
    try:
        mat = materialize_path(tables, nodes, edges)
        num_rows = mat.shape[0]
        num_cols = mat.shape[1]
    except Exception:
        num_rows = 0
        num_cols = 0
    
    hops = len(nodes) - 1
    
    # Generate DOT source with highlighting
    dot_source = path_to_dot_highlighted(
        nodes, edges, tables,
        highlight_blue=highlight_blue,
        highlight_green_nodes=highlight_green_nodes,
    )
    dot_json = json.dumps(dot_source)
    
    # Card styling based on highlighting
    if highlight_blue:
        card_bg = "linear-gradient(135deg, #DBEAFE 0%, #BFDBFE 100%)"
        card_border = "#2563EB"
        badge_bg = "linear-gradient(135deg, #2563EB 0%, #60A5FA 100%)"
    else:
        card_bg = "linear-gradient(135deg, #FFFBEB 0%, #FFF7ED 100%)"
        card_border = "#FED7AA"
        badge_bg = "linear-gradient(135deg, #F97316 0%, #FDBA74 100%)"
    
    # Fixed-height graph area so long join paths grow horizontally without shrinking nodes.
    graph_height_px = 210

    card_html = f"""
    <div class="jp-card" style="background: {card_bg}; border-color: {card_border};">
        <div class="jp-header">
            <span class="jp-rank" style="background: {badge_bg};">#{rank}</span>
            <span class="jp-hops">{hops} hop{'s' if hops != 1 else ''}</span>
            <div class="jp-stats">
                <span class="jp-stat">{num_rows:,} rows</span>
                <span class="jp-stat">{num_cols} cols</span>
            </div>
        </div>
        <div class="jp-graph" id="graph-{rank}"></div>
    </div>
    <script>
        (function() {{
            const dot = {dot_json};
            const viz = new Viz();
            viz.renderSVGElement(dot)
                .then(function(element) {{
                    element.style.height = '{graph_height_px}px';
                    element.style.width = 'auto';
                    element.style.maxWidth = 'none';
                    document.getElementById('graph-{rank}').appendChild(element);
                }})
                .catch(error => {{
                    document.getElementById('graph-{rank}').innerHTML = '<p style="color: red;">Error rendering graph</p>';
                    console.error(error);
                }});
        }})();
    </script>
    """
    return card_html


def render() -> None:
    """Render Stage 3: Retrieved Join Paths with Comparison View and Pagination."""
    if not st.session_state.processed_ready:
        st.info("Go to **Upload and Preprocess** first.")
        if st.button("⬅️ Back", use_container_width=True):
            goto_stage(0)
        return
    if not st.session_state.query_submitted or st.session_state.query_df is None:
        st.info("Go to **Input Query Table** first.")
        if st.button("⬅️ Back", use_container_width=True):
            goto_stage(2)
        return

    tables = st.session_state.tables
    query_df: pd.DataFrame = st.session_state.query_df.reset_index(drop=True)
    st.session_state.query_df = query_df

    max_hops = int(st.session_state.ret_max_hops)
    min_edge_score = float(st.session_state.ret_min_edge_score)
    
    # Initialize pagination and display settings
    if "comparison_page" not in st.session_state:
        st.session_state.comparison_page = 0
    if "num_display_paths" not in st.session_state:
        st.session_state.num_display_paths = 20

    # Default selection: first row
    if st.session_state.get("selected_row") is None and len(query_df) > 0:
        st.session_state.selected_row = 0
    st.session_state.setdefault("selected_column", None)

    selected_row = st.session_state.get("selected_row")
    selected_column = st.session_state.get("selected_column")

    # Build join graph and get paths
    graph = build_join_graph(tables, min_edge_score=min_edge_score)
    graph, tables_with_query = attach_query_node(
        graph=graph, tables=tables, query=query_df, min_edge_score=min_edge_score
    )
    
    # Get all paths (use a higher k initially, then filter)
    all_paths = simple_paths_topk(graph=graph, start="query", max_hops=max_hops, k=50)

    if not all_paths:
        st.warning("No join paths found. Try adjusting preprocessing hyperparameters and re-running preprocessing.")
        return

    # Limit paths to the selected number from session state (slider is on query page)
    num_paths = st.session_state.num_display_paths
    paths = all_paths[:num_paths]
    total_pages = (len(paths) + 1) // 2  # 2 paths per page

    # Ensure current page is valid
    if st.session_state.comparison_page >= total_pages:
        st.session_state.comparison_page = max(0, total_pages - 1)

    current_page = st.session_state.comparison_page

    # ==================== Phase 6: Heatmap Toggle ====================
    st.subheader("Retrieved Join Paths")
    heatmap_on = st.toggle(
        "List view / Heatmap view",
        value=bool(st.session_state.get("ret_heatmap_view", False)),
        key="ret_heatmap_view",
        help="Toggle between comparison view with pagination and heatmap visualization.",
    )
    view_mode = "Heatmap view" if heatmap_on else "List view"

    if view_mode == "Heatmap view":
        # ==================== Heatmap View ====================
        # Compute lightweight path metadata for the heatmap
        cells: List[HeatmapCell] = []
        by_key: Dict[str, Tuple[List[str], float, List[EdgeInfo]]] = {}
        for nodes, score, edges in paths:
            if len(nodes) <= 1:
                continue
            path_key = " -> ".join(nodes)
            by_key[path_key] = (nodes, score, edges)
            try:
                mat = materialize_path(tables=tables_with_query, path_nodes=nodes, path_edges=edges)
                tuple_count = int(mat.shape[0])
            except Exception:
                tuple_count = 0
            cells.append(
                HeatmapCell(
                    path_key=path_key,
                    node_count=len(nodes),
                    tuple_count=tuple_count,
                )
            )

        # Default selection (first path) so the heatmap can highlight it on first render
        if "ret_selected_heatmap_cell" not in st.session_state:
            first = cells[0].path_key if cells else None
            st.session_state.ret_selected_heatmap_cell = (
                {"best_path_key": first, "path_keys": [first]} if first else {}
            )

        left, right = st.columns([0.60, 0.40], gap="large")
        with left:
            sel0 = st.session_state.get("ret_selected_heatmap_cell", {}) or {}
            selected_best_path_key = (
                str(sel0.get("best_path_key"))
                if isinstance(sel0, dict) and sel0.get("best_path_key") is not None
                else None
            )
            picked = render_joinpath_heatmap(
                cells=cells,
                query_param_key="ret_heatmap_sel",
                title="Retrieved join paths heatmap",
                height_px=560,
                selected_best_path_key=selected_best_path_key,
            )

        if isinstance(picked, dict) and picked:
            # Store the clicked heatmap cell selection
            ts = picked.get("ts")
            last_ts = st.session_state.get("ret_heatmap_last_ts")
            st.session_state.ret_selected_heatmap_cell = {k: v for k, v in picked.items() if k != "ts"}

            # Force a single rerun on a *new* click
            if ts is not None and ts != last_ts:
                st.session_state.ret_heatmap_last_ts = ts
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()

        sel = st.session_state.get("ret_selected_heatmap_cell", {}) or {}
        path_keys = sel.get("path_keys") if isinstance(sel, dict) else None
        if not isinstance(path_keys, list):
            path_keys = []
        # Keep only valid join paths
        path_keys = [str(k) for k in path_keys if str(k) in by_key]
        best_key = sel.get("best_path_key") if isinstance(sel, dict) else None
        best_key = str(best_key) if best_key in by_key else (path_keys[0] if path_keys else None)

        with right:
            st.subheader("Selected heatmap cell")
            if not path_keys:
                st.info("Click a heatmap cell to select join paths.")
            else:
                tuple_bin_lo = sel.get("tuple_bin_lo") if isinstance(sel, dict) else None
                tuple_bin_hi = sel.get("tuple_bin_hi") if isinstance(sel, dict) else None
                node_count = sel.get("node_count") if isinstance(sel, dict) else None
                meta_parts = []
                if isinstance(node_count, (int, float)) and int(node_count) > 0:
                    meta_parts.append(f"Nodes: {int(node_count)}")
                if isinstance(tuple_bin_lo, (int, float)) and isinstance(tuple_bin_hi, (int, float)):
                    meta_parts.append(f"Tuple bin: {int(tuple_bin_lo):,}–{int(tuple_bin_hi):,}")
                meta_parts.append(f"Paths in cell: {len(path_keys):,}")
                st.caption(" • ".join(meta_parts))
                st.caption("Expand a path below to view its diagram with highlighting.")

                # Show each path's diagram with highlighting (collapsed behind expanders)
                cell_by_key = {c.path_key: c for c in cells}
                for i, k in enumerate(path_keys):
                    expanded = bool(best_key) and (k == best_key) and len(path_keys) <= 3
                    label = f"{i+1}. {k}"
                    with st.expander(label, expanded=expanded):
                        nodes, score, edges = by_key[k]
                        cmeta = cell_by_key.get(k)
                        if cmeta is not None:
                            st.caption(
                                f"Nodes: {cmeta.node_count} • Tuples: {cmeta.tuple_count:,}"
                            )

                        # Determine highlighting for this path
                        green_nodes = set()
                        if selected_column is not None and selected_column in query_df.columns:
                            green_nodes = get_tables_with_column_value(
                                tables_with_query, selected_column, query_df, selected_row
                            )

                        highlight_blue = False
                        if selected_row is not None:
                            highlight_blue = check_row_match(
                                nodes, edges, tables_with_query, query_df, selected_row
                            )

                        dot_source = path_to_dot_highlighted(
                            nodes, edges, tables=tables_with_query,
                            highlight_blue=highlight_blue,
                            highlight_green_nodes=green_nodes,
                        )
                        render_graphviz_fixed_height(dot_source, height_px=220)

                # Add button to switch to comparison view with these paths
                st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
                if st.button("📊 View in comparison mode", use_container_width=True, type="primary"):
                    # Find the index of the first path in path_keys within the full paths list
                    if path_keys:
                        first_path_key = path_keys[0]
                        nodes_to_find, _, _ = by_key.get(first_path_key, ([], 0, []))
                        if nodes_to_find:
                            # Find this path in the paths list
                            for idx, (nodes, score, edges) in enumerate(paths):
                                if nodes == nodes_to_find:
                                    # Calculate which page this path is on (2 paths per page)
                                    target_page = idx // 2
                                    st.session_state.comparison_page = target_page
                                    break
                    # Switch to comparison view
                    st.session_state.ret_heatmap_view = False
                    st.rerun()

        st.divider()

        # Navigation buttons
        nav_cols = st.columns([0.5, 0.5])
        with nav_cols[0]:
            if st.button("⬅️ Back to Query Input", use_container_width=True):
                goto_stage(2)
        with nav_cols[1]:
            if st.button("🔄 Restart from beginning", type="primary", use_container_width=True):
                goto_stage(0)

        st.stop()

    # ==================== Main Comparison Layout ====================
    # Left column: Query table, Right column: 2 join paths side-by-side
    col_query, col_paths = st.columns([0.35, 0.65], gap="large")

    with col_query:
        st.subheader("Query Table")
        
        # Build table data for the component
        col_names = list(query_df.columns)
        table_data = {
            "columns": [str(c) for c in col_names],
            "rows": [],
        }
        for row_idx in range(len(query_df)):
            row_vals = []
            for col in col_names:
                val = query_df.iloc[row_idx][col]
                row_vals.append("" if pd.isna(val) else str(val))
            table_data["rows"].append(row_vals)

        # Declare the component
        component_dir = os.path.join(os.path.dirname(__file__), "components", "query_table_component")
        query_table_component = components.declare_component("query_table_selector", path=component_dir)

        # Render the component
        result = query_table_component(
            data=table_data,
            selected_row=selected_row,
            selected_column=selected_column,
            key="query_table_phase5",
            default=None,
        )

        # Update session state if selection changed
        if isinstance(result, dict):
            new_row = result.get("selected_row")
            new_col = result.get("selected_column")
            changed = False
            if new_row != selected_row:
                st.session_state.selected_row = new_row
                changed = True
            if new_col != selected_column:
                st.session_state.selected_column = new_col
                changed = True
            if changed:
                st.rerun()

        # Selection legend
        st.markdown("""
        <div style="margin-top: 16px; padding: 12px; background: #F8FAFC; border-radius: 8px; border: 1px solid #E2E8F0;">
            <div style="font-weight: 600; margin-bottom: 8px; color: #334155;">Selection Mode:</div>
            <div style="display: flex; gap: 16px; flex-wrap: wrap;">
                <div style="display: flex; align-items: center; gap: 6px;">
                    <div style="width: 16px; height: 16px; background: #DBEAFE; border: 2px solid #2563EB; border-radius: 3px;"></div>
                    <span style="font-size: 0.9em; color: #475569;">Row → Blue path</span>
                </div>
                <div style="display: flex; align-items: center; gap: 6px;">
                    <div style="width: 16px; height: 16px; background: #D1FAE5; border: 2px solid #059669; border-radius: 3px;"></div>
                    <span style="font-size: 0.9em; color: #475569;">Column → Green nodes</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Actions
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        if st.button("⬅️ Back to Query Input", use_container_width=True):
            goto_stage(2)
        if st.button("🔄 Restart from beginning", type="primary", use_container_width=True):
            goto_stage(0)

    with col_paths:
        st.subheader("Join Path Comparison")
        
        # Pagination controls (just below the subheader)
        nav_cols = st.columns([0.25, 0.5, 0.25])
        with nav_cols[0]:
            if st.button("◀ Previous", use_container_width=True, disabled=current_page == 0):
                st.session_state.comparison_page = max(0, current_page - 1)
                st.rerun()
        with nav_cols[1]:
            st.markdown(
                f"<div style='text-align: center; padding: 6px 0; font-weight: 600; color: #374151;'>Page {current_page + 1} of {total_pages}</div>",
                unsafe_allow_html=True,
            )
        with nav_cols[2]:
            if st.button("Next ▶", use_container_width=True, disabled=current_page >= total_pages - 1):
                st.session_state.comparison_page = min(total_pages - 1, current_page + 1)
                st.rerun()
        
        # Get paths for current page (2 per page)
        start_idx = current_page * 2
        end_idx = min(start_idx + 2, len(paths))
        page_paths = paths[start_idx:end_idx]
        
        # Determine highlighting
        highlight_green_nodes_list = []
        highlight_blue_list = []
        
        for nodes, score, edges in page_paths:
            # Column highlighting (green nodes)
            green_nodes = set()
            if selected_column is not None and selected_column in query_df.columns:
                green_nodes = get_tables_with_column_value(
                    tables_with_query, selected_column, query_df, selected_row
                )
            
            # Row highlighting (blue path)
            highlight_blue = False
            if selected_row is not None:
                highlight_blue = check_row_match(
                    nodes, edges, tables_with_query, query_df, selected_row
                )
            
            highlight_green_nodes_list.append(green_nodes)
            highlight_blue_list.append(highlight_blue)
        
        # Build HTML for both cards
        cards_html = ""
        for i, (path_data, green_nodes, blue) in enumerate(
            zip(page_paths, highlight_green_nodes_list, highlight_blue_list)
        ):
            nodes, score, edges = path_data
            rank = start_idx + i + 1
            cards_html += render_join_path_card(
                rank=rank,
                nodes=nodes,
                edges=edges,
                tables=tables_with_query,
                highlight_blue=blue,
                highlight_green_nodes=green_nodes,
            )
        
        # Full component HTML
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <script src="https://cdn.jsdelivr.net/npm/viz.js@2.1.2/viz.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/viz.js@2.1.2/full.render.js"></script>
            <style>
                * {{
                    box-sizing: border-box;
                }}
                html, body {{
                    margin: 0;
                    padding: 0;
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                    background: transparent;
                }}
                .jp-container {{
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 20px;
                    padding: 8px;
                }}
                .jp-card {{
                    border: 2px solid #FED7AA;
                    border-radius: 12px;
                    padding: 16px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                    transition: transform 0.2s, box-shadow 0.2s;
                }}
                .jp-card:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 4px 12px rgba(0,0,0,0.12);
                }}
                .jp-header {{
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    margin-bottom: 12px;
                    padding-bottom: 12px;
                    border-bottom: 1px solid rgba(0,0,0,0.1);
                    flex-wrap: wrap;
                }}
                .jp-rank {{
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                    padding: 4px 12px;
                    border-radius: 8px;
                    font-weight: 800;
                    font-size: 0.95em;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                }}
                .jp-hops {{
                    display: inline-flex;
                    align-items: center;
                    padding: 4px 10px;
                    background: linear-gradient(135deg, #A78BFA 0%, #C4B5FD 100%);
                    color: white;
                    border-radius: 6px;
                    font-weight: 600;
                    font-size: 0.85em;
                }}
                .jp-stats {{
                    margin-left: auto;
                    display: flex;
                    gap: 8px;
                }}
                .jp-stat {{
                    font-size: 0.85em;
                    color: #475569;
                    background: rgba(255,255,255,0.7);
                    padding: 4px 8px;
                    border-radius: 4px;
                }}
                .jp-graph {{
                    height: 230px;
                    overflow-x: auto;
                    overflow-y: hidden;
                    background: rgba(255,255,255,0.5);
                    border-radius: 8px;
                    padding: 12px;
                }}
                .jp-graph svg {{
                    height: 210px !important;
                    width: auto !important;
                    max-width: none !important;
                    display: block;
                }}
                @media (max-width: 600px) {{
                    .jp-container {{
                        grid-template-columns: 1fr;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="jp-container">
                {cards_html}
            </div>
        </body>
        </html>
        """
        
        # Calculate dynamic height based on content
        # Slightly taller to accommodate fixed-height graph viewport + header.
        base_height = 360
        if len(page_paths) == 1:
            component_height = base_height
        else:
            component_height = base_height
        
        components.html(full_html, height=component_height, scrolling=True)
        
        # Summary info
        st.markdown(f"""
        <div style="margin-top: 12px; padding: 12px 16px; background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%); border-radius: 8px; border: 1px solid #86EFAC;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                <span style="font-weight: 600; color: #166534;">Showing paths {start_idx + 1}-{end_idx} of {len(paths)}</span>
                <span style="font-size: 0.9em; color: #15803D;">Total available: {len(all_paths)} paths</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ==================== Detailed Path View (Expandable) ====================
    st.subheader("All Paths Overview")
    
    # Prepare path data for the table
    rows = []
    for idx, (nodes, score, edges) in enumerate(paths, start=1):
        hops = len(nodes) - 1
        try:
            mat = materialize_path(tables_with_query, nodes, edges)
            num_rows = mat.shape[0]
            num_cols = mat.shape[1]
        except Exception:
            num_rows = 0
            num_cols = 0
        rows.append({
            "Rank": idx,
            "Path": " → ".join(nodes),
            "Hops": hops,
            "Rows": num_rows,
            "Cols": num_cols,
        })
    
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        height=250,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", width="small"),
            "Path": st.column_config.TextColumn("Join Path", width="large"),
            "Hops": st.column_config.NumberColumn("Hops", width="small"),
            "Rows": st.column_config.NumberColumn("Rows", width="small"),
            "Cols": st.column_config.NumberColumn("Cols", width="small"),
        },
    )
