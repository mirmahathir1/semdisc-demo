from __future__ import annotations

import json
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from utils import (
    attach_query_node,
    build_join_graph,
    format_path,
    goto_stage,
    materialize_path,
    path_to_dot,
    simple_paths_topk,
)


def render() -> None:
    """Render Stage 3: Retrieved Join Paths."""
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
    query_df: pd.DataFrame = st.session_state.query_df

    max_hops = int(st.session_state.ret_max_hops)
    min_edge_score = float(st.session_state.ret_min_edge_score)
    top_k = int(st.session_state.ret_top_k)

    graph = build_join_graph(tables, min_edge_score=min_edge_score)
    graph, tables_with_query = attach_query_node(
        graph=graph, tables=tables, query=query_df, min_edge_score=min_edge_score
    )
    paths = simple_paths_topk(graph=graph, start="query", max_hops=max_hops, k=top_k)

    if not paths:
        st.warning("No join paths found. Try adjusting preprocessing hyperparameters and re-running preprocessing.")
        return

    # Prepare path data for the table
    rows = []
    for idx, (nodes, score, edges) in enumerate(paths, start=1):
        hops = len(nodes) - 1
        # Materialize path to get row/column counts
        try:
            mat = materialize_path(
                tables=tables_with_query, path_nodes=nodes, path_edges=edges
            )
            num_rows = mat.shape[0]
            num_cols = mat.shape[1]
        except Exception:
            num_rows = 0
            num_cols = 0
        rows.append(
            {
                "rank": idx,
                "path": " -- ".join(nodes),
                "hops": hops,
                "num_rows": num_rows,
                "num_cols": num_cols,
            }
        )

    # ==================== ROW 1: Top paths list + Dropdown & Materialized table ====================
    col_paths_list, col_materialize = st.columns([0.5, 0.5], gap="large")
    
    with col_paths_list:
        st.subheader("Top join paths")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=350)
    
    with col_materialize:
        labels = [
            f"#{i+1}  {paths[i][0][-1]}  |  {paths[i][0]}"
            for i in range(len(paths))
        ]
        pick = st.selectbox(
            "Select a path to materialize",
            options=list(range(len(paths))),
            format_func=lambda i: labels[i],
        )
        
        selected_nodes, _, selected_edges = paths[pick]
        
        # Materialize the path
        try:
            materialized = materialize_path(
                tables=tables_with_query, path_nodes=selected_nodes, path_edges=selected_edges
            )
        except Exception as e:
            st.error(f"Failed to materialize the selected path: {e}")
            return
        
        st.subheader("Materialized table")
        st.write(f"Rows: {materialized.shape[0]} • Cols: {materialized.shape[1]}")
        st.dataframe(materialized, use_container_width=True, height=250)

    st.divider()

    # ==================== ROW 2: Selected path component + Buttons ====================
    col_path, col_buttons = st.columns([0.7, 0.3], gap="large")
    
    with col_path:
        st.subheader("Selected path")
        
        # Get path metadata
        num_rows = materialized.shape[0]
        num_cols = materialized.shape[1]
        attributes = list(materialized.columns)
        hops = len(selected_nodes) - 1
        rank = pick + 1
        
        # Build attributes list
        attributes_list = ''.join([f'<span>{attr}</span>' for attr in attributes]) if num_cols > 0 else ''
        
        # Generate the DOT source for the graph
        dot_source = path_to_dot(selected_nodes, selected_edges, tables=tables_with_query)
        dot_json = json.dumps(dot_source)
        
        # Stats HTML - display values below labels
        if num_rows > 0 or num_cols > 0:
            stats_html = f'''
            <div class="sd-unified-stats">
                <div class="sd-stat-item">
                    <div class="sd-stat-label">Number of Rows:</div>
                    <div class="sd-stat-value">{num_rows:,}</div>
                </div>
                <div class="sd-stat-item">
                    <div class="sd-stat-label">Number of Columns:</div>
                    <div class="sd-stat-value">{num_cols}</div>
                </div>
                <div class="sd-stat-item">
                    <div class="sd-stat-label">Hops:</div>
                    <div class="sd-stat-value">{hops}</div>
                </div>
            </div>
            '''
        else:
            stats_html = f'''
            <div class="sd-unified-stats">
                <div class="sd-stat-item">
                    <div class="sd-stat-label">Hops:</div>
                    <div class="sd-stat-value">{hops}</div>
                </div>
            </div>
            '''
        
        # Calculate dynamic height based on number of attributes
        attr_rows = max(1, (num_cols + 2) // 3)  # Ceiling division
        content_height = 240 + (attr_rows * 38)
        graph_min_height = 250  # Minimum for graph to display properly
        dynamic_height = max(content_height, graph_min_height) + 40  # Extra buffer
        
        # Create the unified component matching navigation page style
        card_html = f"""
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
                    height: auto;
                    overflow: visible;
                }}
                .sd-unified-component {{
                    padding: 20px 24px;
                    background: linear-gradient(135deg, #FFFBEB 0%, #FFF7ED 100%);
                    border-radius: 12px;
                    border: 2px solid #FED7AA;
                    box-shadow: 0 2px 4px rgba(249, 115, 22, 0.08);
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 24px;
                    min-height: fit-content;
                    height: auto;
                    margin-bottom: 16px;
                }}
                .sd-unified-left {{
                    display: flex;
                    flex-direction: column;
                    min-height: fit-content;
                }}
                .sd-unified-right {{
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    background: #FFFFFF;
                    border-radius: 10px;
                    padding: 16px;
                    border: 1px solid #E2E8F0;
                }}
                .sd-unified-header {{
                    display: flex;
                    align-items: center;
                    gap: 12px;
                    flex-wrap: wrap;
                    margin-bottom: 16px;
                    padding-bottom: 16px;
                    border-bottom: 2px solid #FED7AA;
                }}
                .sd-unified-title {{
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    flex-wrap: wrap;
                }}
                .sd-unified-label {{
                    color: #1E293B;
                    font-weight: 700;
                    font-size: 0.85rem;
                    text-transform: uppercase;
                    letter-spacing: 0.05em;
                }}
                .sd-unified-rank {{
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    background: linear-gradient(135deg, #F97316 0%, #FDBA74 100%);
                    color: white;
                    padding: 4px 12px;
                    border-radius: 8px;
                    font-weight: 800;
                    font-size: 1em;
                    box-shadow: 0 2px 4px rgba(249, 115, 22, 0.3);
                    min-width: 32px;
                }}
                .sd-unified-hops {{
                    display: inline-flex;
                    align-items: center;
                    padding: 6px 14px;
                    background: linear-gradient(135deg, #A78BFA 0%, #C4B5FD 100%);
                    color: white;
                    border-radius: 8px;
                    font-weight: 700;
                    font-size: 0.9rem;
                    box-shadow: 0 2px 4px rgba(167, 139, 250, 0.3);
                }}
                .sd-unified-stats {{
                    display: flex;
                    align-items: flex-start;
                    gap: 24px;
                    margin-left: auto;
                }}
                .sd-stat-item {{
                    display: flex;
                    flex-direction: column;
                    align-items: flex-start;
                    gap: 4px;
                }}
                .sd-stat-label {{
                    font-size: 0.85rem;
                    color: #475569;
                    font-weight: 600;
                    text-transform: uppercase;
                    letter-spacing: 0.05em;
                }}
                .sd-stat-value {{
                    font-size: 1.1rem;
                    color: #334155;
                    font-weight: 700;
                }}
                .sd-unified-attributes {{
                    margin-top: 12px;
                    flex: 1;
                    min-height: fit-content;
                }}
                .sd-unified-attributes-label {{
                    font-size: 0.95rem;
                    color: #334155;
                    font-weight: 700;
                    margin-bottom: 10px;
                    display: block;
                }}
                .sd-unified-attributes-list {{
                    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
                    font-size: 0.88rem;
                    color: #1E293B;
                    font-weight: 500;
                    line-height: 1.8;
                    display: flex;
                    flex-wrap: wrap;
                    gap: 6px;
                }}
                .sd-unified-attributes-list span {{
                    background: linear-gradient(135deg, #F5F3FF 0%, #DDD6FE 100%);
                    padding: 4px 10px;
                    border-radius: 6px;
                    border: 1px solid #C4B5FD;
                    color: #5B21B6;
                    font-weight: 600;
                }}
                #graph {{
                    width: 100%;
                    display: flex;
                    justify-content: center;
                }}
                #graph svg {{
                    max-width: 100%;
                    height: auto;
                }}
            </style>
        </head>
        <body>
            <div class="sd-unified-component">
                <div class="sd-unified-left">
                    <div class="sd-unified-header">
                        <div class="sd-unified-title">
                            <span class="sd-unified-label">Join Path</span>
                            <span class="sd-unified-rank">#{rank}</span>
                        </div>
                        {stats_html}
                    </div>
                    <div class="sd-unified-attributes">
                        <span class="sd-unified-attributes-label">Attributes ({num_cols}):</span>
                        <div class="sd-unified-attributes-list">{attributes_list}</div>
                    </div>
                </div>
                <div class="sd-unified-right">
                    <div id="graph"></div>
                </div>
            </div>
            <script>
                const dot = {dot_json};
                const viz = new Viz();
                viz.renderSVGElement(dot)
                    .then(function(element) {{
                        document.getElementById('graph').appendChild(element);
                    }})
                    .catch(error => {{
                        document.getElementById('graph').innerHTML = '<p style="color: red;">Error rendering graph</p>';
                        console.error(error);
                    }});
            </script>
        </body>
        </html>
        """
        components.html(card_html, height=dynamic_height)
    
    with col_buttons:
        st.subheader("Actions")
        csv_bytes = materialized.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download materialized CSV",
            data=csv_bytes,
            file_name="materialized.csv",
            mime="text/csv",
            use_container_width=True,
        )
        if st.button("⬅️ Back", use_container_width=True):
            goto_stage(2)
        if st.button("🔄 Restart from beginning", type="primary", use_container_width=True):
            goto_stage(0)

