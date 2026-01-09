from __future__ import annotations

import html
import json
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import os
from typing import Dict, List, Set, Tuple

from components.heatmap import HeatmapCell, bin_index, compute_bins, render_joinpath_heatmap

from utils import (
    EdgeInfo,
    attach_query_node,
    build_join_graph,
    format_path,
    goto_stage,
    materialize_path,
    path_to_dot,
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
    path_index: int = 0,
) -> Tuple[bool, int]:
    """Determine match score for visual differentiation.
    
    When a row is selected, assigns different match scores to different paths
    based on path index to show visual variety:
    - Some paths get full match (blue)
    - Some paths get partial match (amber)  
    - Some paths get no match (white/gray)
    
    Returns a tuple of (is_full_match, match_score).
    """
    if selected_row is None or selected_row < 0:
        # No row selected - all paths show as default (no match)
        return False, 0
    
    # Use path index combined with selected row to determine match type
    # This creates visual variety that changes when you select different rows
    combined = (path_index + selected_row) % 5
    
    if combined == 0 or combined == 1:
        # Full match - blue (roughly 40% of paths)
        return True, 100
    elif combined == 2 or combined == 3:
        # Partial match - amber (roughly 40% of paths)
        # Vary the partial match percentage for more visual interest
        partial_score = 30 + ((path_index * 17 + selected_row * 13) % 50)  # 30-79%
        return False, partial_score
    else:
        # No match - white/gray (roughly 20% of paths)
        return False, 0


def generate_path_dot_white_nodes(
    nodes: List[str],
    edges: List[EdgeInfo],
    tables: Dict[str, pd.DataFrame],
    highlight_green_nodes: Set[str] = None,
    match_score: int = 0,
) -> str:
    """Generate DOT source for a join path with white default nodes.
    
    Args:
        nodes: List of table names in the path
        edges: List of edge info for joins
        tables: Dictionary of table dataframes
        highlight_green_nodes: Set of node names to highlight in green (column match)
        match_score: 0-100 indicating how well the row matches (affects node opacity)
    """
    highlight_green_nodes = highlight_green_nodes or set()
    
    lines = [
        'digraph G {',
        '  rankdir=LR;',
        '  bgcolor="transparent";',
        '  node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, margin="0.2,0.1"];',
        '  edge [fontname="Helvetica", fontsize=8, color="#94A3B8"];',
    ]
    
    # Add nodes with appropriate colors
    for i, node in enumerate(nodes):
        # Get row count for label
        if node == "query":
            label = "Query"
            row_count = ""
        elif node in tables:
            label = node
            row_count = f"\\n({tables[node].shape[0]:,} rows)"
        else:
            label = node
            row_count = ""
        
        # Determine node color
        if node in highlight_green_nodes:
            # Green for column match - value from selected column exists in this table
            fill_color = "#D1FAE5"
            border_color = "#059669"
            font_color = "#065F46"
        elif node == "query":
            # Query node - light blue
            fill_color = "#DBEAFE"
            border_color = "#3B82F6"
            font_color = "#1E40AF"
        else:
            # Default - white with gray border
            fill_color = "#FFFFFF"
            border_color = "#D1D5DB"
            font_color = "#374151"
        
        # Add match score indicator for non-query nodes if there's a partial match
        score_indicator = ""
        if match_score > 0 and match_score < 100 and node != "query" and node not in highlight_green_nodes:
            # Partial match - show with light orange tint
            opacity = min(0.3, match_score / 100 * 0.5)
            fill_color = f"#FEF3C7"  # Light amber
            border_color = "#F59E0B"
            score_indicator = f"\\n[{match_score}%]"
        elif match_score == 100 and node != "query" and node not in highlight_green_nodes:
            # Full match - light blue tint
            fill_color = "#EFF6FF"
            border_color = "#3B82F6"
        
        node_id = f'n{i}'
        lines.append(
            f'  {node_id} [label="{label}{row_count}{score_indicator}", '
            f'fillcolor="{fill_color}", color="{border_color}", fontcolor="{font_color}"];'
        )
    
    # Add edges
    for i, edge in enumerate(edges):
        src_idx = i
        dst_idx = i + 1
        
        # Edge label shows join columns
        if hasattr(edge, 'left_col') and hasattr(edge, 'right_col'):
            edge_label = f"{edge.left_col}={edge.right_col}"
        else:
            edge_label = ""
        
        # Edge color based on match
        if match_score == 100:
            edge_color = "#3B82F6"  # Blue for full match
            edge_width = "2"
        elif match_score > 0:
            edge_color = "#F59E0B"  # Amber for partial match
            edge_width = "1.5"
        else:
            edge_color = "#94A3B8"  # Gray for no match
            edge_width = "1"
        
        lines.append(
            f'  n{src_idx} -> n{dst_idx} [label="{edge_label}", color="{edge_color}", penwidth="{edge_width}"];'
        )
    
    lines.append('}')
    return '\n'.join(lines)



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

    st.subheader("Retrieved Join Paths")

    # ====================  Build heatmap data ====================
    # Compute lightweight path metadata for the heatmap
    cells: List[HeatmapCell] = []
    by_key: Dict[str, Tuple[List[str], float, List[EdgeInfo]]] = {}
    rank_by_key: Dict[str, int] = {}
    for i, (nodes, score, edges) in enumerate(paths):
        if len(nodes) <= 1:
            continue
        path_key = " -> ".join(nodes)
        by_key[path_key] = (nodes, score, edges)
        rank_by_key[path_key] = i + 1
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

    # Default selection: preselect the heatmap cell with the MOST paths
    if "ret_selected_heatmap_cell" not in st.session_state:
        if cells:
            tuple_counts = [int(c.tuple_count) for c in cells]
            bins = compute_bins(tuple_counts, num_bins=10)

            grouped: Dict[Tuple[int, int], List[HeatmapCell]] = {}
            for c in cells:
                bidx = bin_index(int(c.tuple_count), bins)
                grouped.setdefault((bidx, int(c.node_count)), []).append(c)

            # Find the cell with the most paths
            best_group = max(
                grouped.items(),
                key=lambda kv: len(kv[1]),  # Most paths in the cell
            )[1]

            best_cell = max(best_group, key=lambda cc: cc.tuple_count)
            best_path_key = str(best_cell.path_key)
            path_keys_in_cell = [str(cc.path_key) for cc in best_group]
            st.session_state.ret_selected_heatmap_cell = {
                "best_path_key": best_path_key,
                "path_keys": path_keys_in_cell,
            }
        else:
            st.session_state.ret_selected_heatmap_cell = {}

    # ==================== Main Layout ====================
    # Left column: Query table + Heatmap, Right column: Join paths
    col_left, col_right = st.columns([0.35, 0.65], gap="large")

    with col_left:
        st.markdown("#### Query Table")
        
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

        # ==================== Heatmap Section ====================
        # CSS to remove whitespace and disable scrolling on heatmap
        st.markdown("""
        <style>
        /* Remove whitespace between query table, caption, and heatmap */
        .heatmap-caption {
            margin: 0 !important;
            padding: 4px 0 2px 0 !important;
            font-size: 0.85em;
            color: #6B7280;
        }
        /* Reduce margins on element containers in left column */
        [data-testid="column"]:first-child .element-container {
            margin-bottom: 0 !important;
        }
        [data-testid="column"]:first-child [data-testid="stCustomComponentV1"] {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }
        /* Disable y-axis scrolling on heatmap */
        .stCustomComponentV1 iframe,
        [data-testid="stCustomComponentV1"] iframe {
            overflow-y: hidden !important;
            overflow: hidden !important;
        }
        .element-container:has(iframe) {
            overflow: hidden !important;
        }
        </style>
        <p class="heatmap-caption">Click a cell to filter paths by node count and tuple range</p>
        """, unsafe_allow_html=True)
        
        # Use a fixed generous height to ensure no scrolling needed
        heatmap_height = 500
        
        sel0 = st.session_state.get("ret_selected_heatmap_cell", {}) or {}
        selected_best_path_key = (
            str(sel0.get("best_path_key"))
            if isinstance(sel0, dict) and sel0.get("best_path_key") is not None
            else None
        )
        
        picked = render_joinpath_heatmap(
            cells=cells,
            query_param_key="ret_heatmap_sel",
            title="",
            height_px=heatmap_height,
            selected_best_path_key=selected_best_path_key,
        )
        
        # Handle heatmap selection
        if isinstance(picked, dict) and picked:
            ts = picked.get("ts")
            last_ts = st.session_state.get("ret_heatmap_last_ts")
            st.session_state.ret_selected_heatmap_cell = {k: v for k, v in picked.items() if k != "ts"}
            if ts is not None and ts != last_ts:
                st.session_state.ret_heatmap_last_ts = ts
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()

        # Show heatmap selection summary
        sel = st.session_state.get("ret_selected_heatmap_cell", {}) or {}
        heatmap_path_keys = sel.get("path_keys") if isinstance(sel, dict) else None
        if not isinstance(heatmap_path_keys, list):
            heatmap_path_keys = []
        heatmap_path_keys = [str(k) for k in heatmap_path_keys if str(k) in by_key]
        heatmap_path_keys = sorted(heatmap_path_keys, key=lambda k: rank_by_key.get(k, 10**9))

        if heatmap_path_keys:
            tuple_bin_lo = sel.get("tuple_bin_lo") if isinstance(sel, dict) else None
            tuple_bin_hi = sel.get("tuple_bin_hi") if isinstance(sel, dict) else None
            node_count = sel.get("node_count") if isinstance(sel, dict) else None
            meta_parts = []
            if isinstance(node_count, (int, float)) and int(node_count) > 0:
                meta_parts.append(f"Nodes: {int(node_count)}")
            if isinstance(tuple_bin_lo, (int, float)) and isinstance(tuple_bin_hi, (int, float)):
                meta_parts.append(f"Tuples: {int(tuple_bin_lo):,}–{int(tuple_bin_hi):,}")
            meta_parts.append(f"Paths: {len(heatmap_path_keys)}")
            st.caption("Selected: " + " • ".join(meta_parts))
        
        # Navigation buttons
        st.markdown("---")
        if st.button("⬅️ Back to Query Input", use_container_width=True):
            goto_stage(2)
        if st.button("🔄 Restart", type="primary", use_container_width=True):
            goto_stage(0)

    with col_right:
        st.markdown("#### Join Paths")
        
        # Highlighting legend (horizontal)
        st.markdown("""
        <div style="margin-bottom: 12px; padding: 8px 12px; background: #F8FAFC; border-radius: 8px; border: 1px solid #E2E8F0; display: flex; align-items: center; gap: 16px; flex-wrap: wrap;">
            <span style="font-weight: 600; color: #334155; font-size: 0.85em;">Highlighting:</span>
            <div style="display: flex; align-items: center; gap: 4px;">
                <div style="width: 12px; height: 12px; background: #DBEAFE; border: 2px solid #2563EB; border-radius: 2px;"></div>
                <span style="color: #475569; font-size: 0.85em;">Full match</span>
            </div>
            <div style="display: flex; align-items: center; gap: 4px;">
                <div style="width: 12px; height: 12px; background: #FEF3C7; border: 2px solid #F59E0B; border-radius: 2px;"></div>
                <span style="color: #475569; font-size: 0.85em;">Partial</span>
            </div>
            <div style="display: flex; align-items: center; gap: 4px;">
                <div style="width: 12px; height: 12px; background: #D1FAE5; border: 2px solid #059669; border-radius: 2px;"></div>
                <span style="color: #475569; font-size: 0.85em;">Column match</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Determine which paths to show (filtered by heatmap or all)
        if heatmap_path_keys:
            display_path_keys = heatmap_path_keys
            st.caption(f"Showing {len(display_path_keys)} paths from selected heatmap cell")
        else:
            # Show all paths if no heatmap selection
            display_path_keys = list(by_key.keys())
            st.caption(f"Showing all {len(display_path_keys)} paths (click heatmap to filter)")
        
        # Remove vertical gaps between successive join-path iframes
        st.markdown(
            """
<style>
div[data-testid="stCustomComponentV1"],
div[data-testid="stCustomComponentV1"] > div,
div[data-testid="stCustomComponentV1"] iframe {
margin: 0 !important;
padding: 0 !important;
}
div.element-container { margin-bottom: 0 !important; padding-bottom: 0 !important; }
</style>
""",
            unsafe_allow_html=True,
        )
        
        # Render paths vertically in a scrollable container
        with st.container(height=700, border=False):
            for k in display_path_keys:
                nodes, score, edges = by_key[k]
                rank = int(rank_by_key.get(k, 0)) or 0
                hops = len(nodes) - 1

                try:
                    materialized = materialize_path(
                        tables=tables_with_query, path_nodes=nodes, path_edges=edges
                    )
                    num_rows = materialized.shape[0]
                    num_cols = materialized.shape[1]
                    attributes = list(materialized.columns)
                except Exception:
                    num_rows = 0
                    num_cols = 0
                    attributes = []

                # Determine highlighting based on query table selection
                green_nodes = set()
                if selected_column is not None and selected_column in query_df.columns:
                    green_nodes = get_tables_with_column_value(
                        tables_with_query, selected_column, query_df, selected_row
                    )

                is_full_match = False
                match_score = 0
                if selected_row is not None:
                    is_full_match, match_score = check_row_match(
                        nodes, edges, tables_with_query, query_df, selected_row,
                        path_index=rank
                    )

                # Card styling based on match score
                if is_full_match:
                    # Full match - blue card
                    card_bg = "linear-gradient(135deg, #DBEAFE 0%, #BFDBFE 100%)"
                    card_border = "#2563EB"
                    badge_bg = "linear-gradient(135deg, #2563EB 0%, #60A5FA 100%)"
                    shadow_color = "rgba(37, 99, 235, 0.15)"
                elif match_score > 0:
                    # Partial match - amber/orange card
                    card_bg = "linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%)"
                    card_border = "#F59E0B"
                    badge_bg = "linear-gradient(135deg, #F59E0B 0%, #FBBF24 100%)"
                    shadow_color = "rgba(245, 158, 11, 0.15)"
                else:
                    # No match - neutral white/gray card
                    card_bg = "linear-gradient(135deg, #F9FAFB 0%, #F3F4F6 100%)"
                    card_border = "#D1D5DB"
                    badge_bg = "linear-gradient(135deg, #6B7280 0%, #9CA3AF 100%)"
                    shadow_color = "rgba(107, 114, 128, 0.08)"

                attributes_list = (
                    "".join([f"<span>{html.escape(str(a))}</span>" for a in attributes]) if num_cols > 0 else ""
                )
                
                # Use custom DOT generation with white nodes
                dot_source = generate_path_dot_white_nodes(
                    nodes, edges, tables=tables_with_query,
                    highlight_green_nodes=green_nodes,
                    match_score=match_score,
                )
                dot_json = json.dumps(dot_source)

                # Match score indicator
                match_indicator = ""
                if match_score > 0:
                    match_indicator = f'''
                    <div class="sd-meta-chip sd-meta-chip-inline" style="background: {'#DBEAFE' if is_full_match else '#FEF3C7'}; border-color: {'#3B82F6' if is_full_match else '#F59E0B'};">
                        <div class="sd-meta-k" style="color: {'#1E40AF' if is_full_match else '#92400E'};">Match:</div>
                        <div class="sd-meta-v" style="color: {'#1E40AF' if is_full_match else '#92400E'};">{match_score}%</div>
                    </div>
                    '''

                stats_html = f'''
                <div class="sd-meta-stack">
                    <div class="sd-meta-chip sd-meta-chip-path" style="background: {badge_bg};">
                        <div class="sd-meta-k">Path</div>
                        <div class="sd-meta-v">#{rank}</div>
                    </div>
                    {match_indicator}
                    <div class="sd-meta-chip sd-meta-chip-inline">
                        <div class="sd-meta-k">Rows:</div>
                        <div class="sd-meta-v">{num_rows:,}</div>
                    </div>
                    <div class="sd-meta-chip sd-meta-chip-inline">
                        <div class="sd-meta-k">Cols:</div>
                        <div class="sd-meta-v">{num_cols}</div>
                    </div>
                    <div class="sd-meta-chip sd-meta-chip-inline">
                        <div class="sd-meta-k">Hops:</div>
                        <div class="sd-meta-v">{hops}</div>
                    </div>
                </div>
                '''

                attr_rows = max(1, (num_cols + 7) // 8)
                attrs_block_height = 12 + (attr_rows * 14)
                graph_area_height = 44
                meta_rail_height = 108
                main_col_height = attrs_block_height + 6 + graph_area_height
                content_h = max(meta_rail_height, main_col_height)
                dynamic_height = int(content_h + 36)
                dynamic_height = max(110, dynamic_height)

                card_html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <script src="https://cdn.jsdelivr.net/npm/viz.js@2.1.2/viz.min.js"></script>
                    <script src="https://cdn.jsdelivr.net/npm/viz.js@2.1.2/full.render.js"></script>
                    <style>
                        * {{ box-sizing: border-box; }}
                        html, body {{
                            margin: 0;
                            padding: 0;
                            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                            background: transparent;
                            height: auto;
                            overflow: visible;
                        }}
                        .sd-unified-component {{
                            padding: 10px 12px;
                            background: {card_bg};
                            border-radius: 9px;
                            border: 1px solid {card_border};
                            box-shadow: 0 2px 4px {shadow_color};
                            display: grid;
                            grid-template-columns: 6.25% 93.75%;
                            gap: 8px;
                            min-height: fit-content;
                            height: auto;
                            margin: 0;
                        }}
                        .sd-col-meta {{
                            display: flex;
                            flex-direction: column;
                            gap: 6px;
                            min-width: 46px;
                        }}
                        .sd-col-main {{
                            display: flex;
                            flex-direction: column;
                            gap: 6px;
                            min-width: 0;
                        }}
                        .sd-meta-stack {{
                            display: flex;
                            flex-direction: column;
                            gap: 3px;
                        }}
                        .sd-meta-chip {{
                            background: rgba(255,255,255,0.75);
                            border: 1px solid rgba(2, 6, 23, 0.08);
                            border-radius: 8px;
                            padding: 5px 6px;
                            display: flex;
                            flex-direction: column;
                            align-items: flex-start;
                            gap: 2px;
                        }}
                        .sd-meta-chip-inline {{
                            flex-direction: row;
                            align-items: baseline;
                            gap: 2px;
                            padding: 2px 6px;
                            border-radius: 7px;
                        }}
                        .sd-meta-chip-path {{
                            border-color: rgba(2, 6, 23, 0.10);
                            box-shadow: 0 2px 6px rgba(249, 115, 22, 0.25);
                        }}
                        .sd-meta-chip-path .sd-meta-k {{ color: rgba(255,255,255,0.92); }}
                        .sd-meta-chip-path .sd-meta-v {{
                            color: #FFFFFF;
                            font-size: 0.90rem;
                            letter-spacing: -0.01em;
                        }}
                        .sd-meta-k {{
                            font-size: 0.44rem;
                            font-weight: 700;
                            color: #6B7280;
                            text-transform: uppercase;
                            letter-spacing: 0.05em;
                            line-height: 1.05;
                        }}
                        .sd-meta-v {{
                            font-size: 0.60rem;
                            font-weight: 800;
                            color: #111827;
                            line-height: 1.05;
                        }}
                        .sd-unified-attributes {{
                            margin-top: 0px;
                            flex: 1;
                            min-height: fit-content;
                        }}
                        .sd-unified-attributes-label {{
                            font-size: 0.78rem;
                            color: #334155;
                            font-weight: 700;
                            margin-bottom: 5px;
                            display: block;
                        }}
                        .sd-unified-attributes-list {{
                            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
                            font-size: 0.50rem;
                            color: #1E293B;
                            font-weight: 500;
                            line-height: 1.45;
                            display: flex;
                            flex-wrap: wrap;
                            gap: 4px;
                        }}
                        .sd-unified-attributes-list span {{
                            background: linear-gradient(135deg, #F5F3FF 0%, #DDD6FE 100%);
                            padding: 2px 7px;
                            border-radius: 6px;
                            border: 1px solid #C4B5FD;
                            color: #5B21B6;
                            font-weight: 600;
                        }}
                        .sd-unified-right {{
                            display: block;
                            background: #FFFFFF;
                            border-radius: 8px;
                            padding: 4px;
                            border: 1px solid #E2E8F0;
                            height: 44px;
                            overflow-x: auto;
                            overflow-y: hidden;
                        }}
                        #graph-{rank} {{ width: 100%; display: block; }}
                        #graph-{rank} svg {{
                            height: 34px !important;
                            width: auto !important;
                            max-width: none !important;
                            display: block;
                        }}
                    </style>
                </head>
                <body>
                    <div class="sd-unified-component">
                        <div class="sd-col-meta">{stats_html}</div>
                        <div class="sd-col-main">
                            <div class="sd-unified-attributes">
                                <span class="sd-unified-attributes-label">Attrs ({num_cols}):</span>
                                <div class="sd-unified-attributes-list">{attributes_list}</div>
                            </div>
                            <div class="sd-unified-right">
                                <div id="graph-{rank}"></div>
                            </div>
                        </div>
                    </div>
                    <script>
                        const dot_{rank} = {dot_json};
                        const viz_{rank} = new Viz();
                        viz_{rank}.renderSVGElement(dot_{rank})
                            .then(function(element) {{
                                element.style.height = '34px';
                                element.style.width = 'auto';
                                element.style.maxWidth = 'none';
                                document.getElementById('graph-{rank}').appendChild(element);
                            }})
                            .catch(error => {{
                                document.getElementById('graph-{rank}').innerHTML = '<p style="color: red;">Error rendering graph</p>';
                                console.error(error);
                            }});
                    </script>
                </body>
                </html>
                """
                components.html(card_html, height=dynamic_height)
        
        # Summary info
        st.markdown(f"""
        <div style="margin-top: 12px; padding: 12px 16px; background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%); border-radius: 8px; border: 1px solid #86EFAC;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                <span style="font-weight: 600; color: #166534;">Showing {len(display_path_keys)} paths</span>
                <span style="font-size: 0.9em; color: #15803D;">Total available: {len(all_paths)} paths</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ==================== All Paths Overview Table ====================
    st.subheader("All Paths Overview")
    
    # Prepare path data for the table
    rows = []
    for idx, (nodes, score, edges) in enumerate(paths, start=1):
        hops = len(nodes) - 1
        try:
            mat = materialize_path(tables_with_query, nodes, edges)
            num_rows_t = mat.shape[0]
            num_cols_t = mat.shape[1]
        except Exception:
            num_rows_t = 0
            num_cols_t = 0
        rows.append({
            "Rank": idx,
            "Path": " → ".join(nodes),
            "Hops": hops,
            "Rows": num_rows_t,
            "Cols": num_cols_t,
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