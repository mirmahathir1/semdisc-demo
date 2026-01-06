from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

from utils import (
    EdgeInfo,
    build_join_graph,
    drop_fully_blank_rows,
    goto_stage,
    path_to_dot,
    simple_paths_topk,
)


def render() -> None:
    """Render Stage 1: Navigate Processed Data."""
    if not st.session_state.processed_ready:
        st.info("Go to **Upload and Preprocess** first.")
        c_nav1, c_nav2 = st.columns([0.45, 0.55])
        with c_nav1:
            if st.button("Back to previous stage", use_container_width=True):
                goto_stage(0)
        with c_nav2:
            if st.button("Proceed to next stage", type="primary", use_container_width=True):
                st.warning("Please run **Preprocess** before proceeding.")
        return

    tables: Dict[str, pd.DataFrame] = st.session_state.tables
    nav_min_edge_score = float(st.session_state.nav_min_edge_score)
    nav_max_hops = int(st.session_state.nav_max_hops)
    nav_top_k_per_table = 10

    graph = build_join_graph(tables, min_edge_score=nav_min_edge_score)

    # Compute "all" join paths by enumerating top-k paths from every start table,
    # then deduplicate and globally sort by score.
    paths: List[Tuple[List[str], float, List[EdgeInfo]]] = []
    if graph:
        by_nodes: Dict[Tuple[str, ...], Tuple[List[str], float, List[EdgeInfo]]] = {}
        for start_table in sorted(tables.keys()):
            for nodes, score, edges in simple_paths_topk(
                graph=graph,
                start=start_table,
                max_hops=nav_max_hops,
                k=nav_top_k_per_table,
            ):
                if len(nodes) <= 1:
                    continue
                key = tuple(nodes)
                prev = by_nodes.get(key)
                if prev is None or score > prev[1]:
                    by_nodes[key] = (nodes, score, edges)
        paths = sorted(by_nodes.values(), key=lambda x: x[1], reverse=True)

    top_left, top_right = st.columns([0.52, 0.48], gap="large")

    with top_left:
        st.subheader("Table preview")
        table_name = st.selectbox("Table", options=sorted(tables.keys()))
        df = drop_fully_blank_rows(tables[table_name])
        st.write(f"Rows: {df.shape[0]} • Cols: {df.shape[1]}")
        # Show ~5 rows worth of height; remaining rows are scrollable within the table.
        header_px = 36
        row_px = 35
        preview_rows = 5
        table_height = header_px + row_px * max(1, min(preview_rows, int(df.shape[0])))
        st.dataframe(df, use_container_width=True, height=table_height)

    with top_right:
        st.subheader("Join paths")
        if not graph:
            st.warning("No join edges found with the current settings.")
        elif not paths:
            st.warning("No join paths found with the current settings.")
        else:
            st.markdown(
                """
<style>
  .sd-pathcard { border: 1px solid #E5E7EB; background: #FFFFFF; border-radius: 12px; padding: 10px 12px; margin-bottom: 10px; }
  .sd-pathhead { display: flex; justify-content: space-between; gap: 12px; align-items: baseline; }
  .sd-rank { font-weight: 750; color: #111827; }
  .sd-score { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; color: #111827; }
  .sd-meta { font-size: 0.86rem; color: #374151; margin-top: 6px; }
  .sd-path { margin-top: 6px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size: 0.86rem; color: #111827; background: #F9FAFB; border: 1px solid #F3F4F6; border-radius: 10px; padding: 8px; }
  .sd-pathgraph-container { height: 320px; overflow: auto; margin-top: 0; }
  .sd-pathgraph-container div[data-testid="stGraphVizChart"] { height: 320px; overflow: auto; }
  .sd-pathgraph-container div[data-testid="stGraphVizChart"] svg { height: 320px !important; width: auto !important; }
</style>
""",
                unsafe_allow_html=True,
            )

            # Render a per-path button + HTML card in a loop (no dropdown).
            st.session_state.setdefault("nav_selected_path_idx", 0)
            selected_idx = int(st.session_state.nav_selected_path_idx or 0)
            selected_idx = max(0, min(selected_idx, len(paths) - 1))
            st.session_state.nav_selected_path_idx = selected_idx

            # CSS for styling path cards
            st.markdown(
                """
<style>
  .sd-pathcard { border: 1px solid #E5E7EB; background: #FFFFFF; border-radius: 12px; padding: 10px 12px; margin-bottom: 10px; }
  .sd-pathhead { display: flex; justify-content: space-between; gap: 12px; align-items: baseline; }
  .sd-rank { font-weight: 750; color: #111827; }
  .sd-score { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; color: #111827; }
  .sd-meta { font-size: 0.86rem; color: #374151; margin-top: 6px; }
  .sd-path { margin-top: 6px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size: 0.86rem; color: #111827; background: #F9FAFB; border: 1px solid #F3F4F6; border-radius: 10px; padding: 8px; }
</style>
""",
                unsafe_allow_html=True,
            )
            
            # Use Streamlit's native scrollable container (height parameter)
            with st.container(height=560, border=True):
                for idx, (nodes, score, edges) in enumerate(paths):
                    if len(nodes) <= 1:
                        continue
                    rank = idx + 1
                    hops = len(nodes) - 1
                    join_keys = ", ".join([e.key for e in edges]) if edges else ""
                    path_txt = " -> ".join(nodes)
                    
                    bcol, ccol = st.columns([0.22, 0.78], vertical_alignment="top")
                    with bcol:
                        label = "Showing" if idx == selected_idx else "Show join path"
                        if st.button(label, key=f"nav_show_join_path_{rank}", use_container_width=True):
                            st.session_state.nav_selected_path_idx = idx
                            st.rerun()
                    with ccol:
                        card_bg = "#EFF6FF" if idx == selected_idx else "#FFFFFF"
                        card_html = "\n".join(
                            [
                                f'<div class="sd-pathcard" style="background: {card_bg};">',
                                '<div class="sd-pathhead">',
                                f'<div class="sd-rank">#{rank} • {nodes[0]} → {nodes[-1]} • {hops} hop{"s" if hops != 1 else ""}</div>',
                                f'<div class="sd-score">{float(score):.4f}</div>',
                                "</div>",
                                f'<div class="sd-meta"><b>Join keys:</b> {join_keys if join_keys else "—"}</div>',
                                f'<div class="sd-path">{path_txt}</div>',
                                "</div>",
                            ]
                        )
                        st.markdown(card_html, unsafe_allow_html=True)
                    
                    # Render join path graph inline right below the selected card
                    if idx == selected_idx:
                        st.graphviz_chart(path_to_dot(nodes, edges), use_container_width=True)

    st.divider()
    if st.button("Proceed to next stage", type="primary", use_container_width=True):
        goto_stage(2)

