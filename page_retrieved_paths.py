from __future__ import annotations

import pandas as pd
import streamlit as st

from utils import (
    attach_query_node,
    build_join_graph,
    format_path,
    goto_stage,
    materialize_path,
    path_to_dot,
    simple_paths_topk,
    to_dot,
)


def render() -> None:
    """Render Stage 3: Retrieved Join Paths."""
    if not st.session_state.processed_ready:
        st.info("Go to **Upload and Preprocess** first.")
        if st.button("Back to previous stage", use_container_width=True):
            goto_stage(0)
        return
    if not st.session_state.query_submitted or st.session_state.query_df is None:
        st.info("Go to **Input Query Table** first.")
        if st.button("Back to previous stage", use_container_width=True):
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

    left, right = st.columns([0.55, 0.45], gap="large")
    
    with left:
        st.subheader("Top join paths")
        if not paths:
            st.warning("No join paths found. Try adjusting preprocessing hyperparameters and re-running preprocessing.")
            return

        rows = []
        for idx, (nodes, score, edges) in enumerate(paths, start=1):
            rows.append(
                {
                    "rank": idx,
                    "score": round(float(score), 4),
                    "path": " -> ".join(nodes),
                    "joins": ", ".join([e.key for e in edges]) if edges else "",
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=240)

        labels = [
            f"#{i+1}  {paths[i][1]:.4f}  |  {paths[i][0][-1]}  |  {paths[i][0]}"
            for i in range(len(paths))
        ]
        pick = st.selectbox(
            "Select a path to materialize",
            options=list(range(len(paths))),
            format_func=lambda i: labels[i],
        )

    selected_nodes, selected_score, selected_edges = paths[pick]

    with right:
        st.subheader("Highlighted path in join graph")
        highlight = [(e.left, e.right) for e in selected_edges]
        st.graphviz_chart(to_dot(graph, highlight_edges=highlight), use_container_width=True)

    st.divider()
    
    # Materialize the path first (needed for both columns)
    try:
        materialized = materialize_path(
            tables=tables_with_query, path_nodes=selected_nodes, path_edges=selected_edges
        )
    except Exception as e:
        st.error(f"Failed to materialize the selected path: {e}")
        return
    
    # Display materialized table and selected path side by side
    col_table, col_path = st.columns([0.5, 0.5], gap="large")
    
    with col_table:
        st.subheader("Materialized table")
        st.write(f"Rows: {materialized.shape[0]} • Cols: {materialized.shape[1]}")
        # Height set to show approximately 5 rows initially, with scrolling for the rest
        st.dataframe(materialized, use_container_width=True, height=220)
    
    with col_path:
        st.subheader("Selected path")
        st.write(f"**Score:** {selected_score:.4f}")
        st.graphviz_chart(path_to_dot(selected_nodes, selected_edges), use_container_width=True)
    
    st.divider()
    csv_bytes = materialized.to_csv(index=False).encode("utf-8")
    c_nav1, c_nav2, c_nav3 = st.columns([0.33, 0.33, 0.34])
    with c_nav1:
        st.download_button(
            "Download materialized CSV",
            data=csv_bytes,
            file_name="materialized.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c_nav2:
        if st.button("Back to previous stage", use_container_width=True):
            goto_stage(2)
    with c_nav3:
        if st.button("Proceed to next stage (restart)", type="primary", use_container_width=True):
            goto_stage(0)

