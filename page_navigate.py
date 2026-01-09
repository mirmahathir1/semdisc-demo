from __future__ import annotations

import json
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from components.heatmap import HeatmapCell, bin_index, compute_bins, render_joinpath_heatmap

from utils import (
    EdgeInfo,
    build_join_graph,
    goto_stage,
    materialize_path,
    path_to_dot,
    render_graphviz_fixed_height,
    simple_paths_topk,
    to_dot,
    SEMANTIC_TYPES,
)


def _semantic_types_rows(tables: Dict[str, pd.DataFrame], table_name: str) -> pd.DataFrame:
    """Get semantic types for a specific table."""
    rows = []
    if table_name in tables:
        df = tables[table_name]
        for col in list(df.columns):
            stype = SEMANTIC_TYPES.get((table_name, col), "Unknown")
            rows.append(
                {
                    "Column": col,
                    "Semantic Type": stype,
                }
            )
    return pd.DataFrame(rows)


def _semantic_types_badge_table(df: pd.DataFrame) -> str:
    """Render a compact HTML table with semantic type badges."""
    def esc(x: object) -> str:
        return (
            str(x)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    def badge_color(text: str) -> str:
        import hashlib
        h = hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()
        hue = int(h[:2], 16) * 360 // 255
        return f"hsl({hue}, 70%, 92%)"

    head = """
<style>
  .sd-semtypes-compact {
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    overflow: hidden;
    background: white;
    font-size: 0.82rem;
  }
  .sd-semtypes-compact table { width: 100%; border-collapse: collapse; }
  .sd-semtypes-compact thead th {
    background: linear-gradient(135deg, #FFFBEB 0%, #FFF7ED 100%);
    border-bottom: 1px solid #E5E7EB;
    text-align: left;
    padding: 6px 8px;
    font-weight: 700;
    color: #111827;
    font-size: 0.80rem;
  }
  .sd-semtypes-compact tbody td {
    border-bottom: 1px solid #F3F4F6;
    padding: 5px 8px;
    vertical-align: middle;
    font-size: 0.80rem;
    color: #111827;
  }
  .sd-semtypes-compact tbody tr:hover td { background: #FAFAFA; }
  .sd-pill-compact {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 999px;
    border: 1px solid rgba(2, 6, 23, 0.10);
    font-weight: 600;
    color: #0F172A;
    white-space: nowrap;
    font-size: 0.75rem;
  }
  .sd-mono-compact {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    color: #334155;
    font-weight: 600;
    font-size: 0.78rem;
  }
</style>
""".strip()

    body = ['<div class="sd-semtypes-compact"><table>']
    body.append("<thead><tr><th>Column</th><th>Semantic Type</th></tr></thead>")
    body.append("<tbody>")
    for _, r in df.iterrows():
        col = esc(r["Column"])
        stype = str(r["Semantic Type"])
        color = badge_color(stype)
        body.append(
            "<tr>"
            f'<td class="sd-mono-compact">{col}</td>'
            f'<td><span class="sd-pill-compact" style="background:{color};">{esc(stype)}</span></td>'
            "</tr>"
        )
    body.append("</tbody></table></div>")
    return head + "\n" + "\n".join(body)


def render() -> None:
    """Render Stage 1: Navigate Processed Data."""
    if not st.session_state.processed_ready:
        st.info("Go to **Upload and Preprocess** first.")
        c_nav1, c_nav2 = st.columns([0.45, 0.55])
        with c_nav1:
            if st.button("⬅️ Back", use_container_width=True):
                goto_stage(0)
        with c_nav2:
            if st.button("➡️ Proceed to next stage", type="primary", use_container_width=True):
                st.warning("Please run **Preprocess** before proceeding.")
        return

    tables: Dict[str, pd.DataFrame] = st.session_state.tables
    nav_min_edge_score = float(st.session_state.nav_min_edge_score)
    nav_max_hops = int(st.session_state.nav_max_hops)
    nav_top_k_per_table = 10

    graph = build_join_graph(tables, min_edge_score=nav_min_edge_score)

    # Removed semantic types tab - now it's under heatmap
    tab_join_paths, tab_join_graph, tab_tables = st.tabs(
        ["Join paths", "Join graph", "Tables"]
    )

    with tab_join_graph:
        st.subheader("Join graph")
        st.caption("Undirected graph inferred from shared column names + overlapping values (Jaccard).")
        if not graph:
            st.warning("No join edges found with the current settings.")
        else:
            st.graphviz_chart(to_dot(graph), use_container_width=True)

    with tab_tables:
        st.subheader("Tables")
        st.caption("Browse the processed data lake tables.")

        if not tables:
            st.warning("No tables available.")
        else:
            table_name = st.selectbox("Table", options=sorted(tables.keys()), key="nav_table_preview")
            df = tables[table_name]
            st.write(f"Rows: {df.shape[0]:,} • Cols: {df.shape[1]}")
            st.dataframe(df, use_container_width=True, height=520)

    with tab_join_paths:
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

        if not graph:
            st.warning("No join edges found with the current settings.")
            return
        if not paths:
            st.warning("No join paths found with the current settings.")
            return

        # Phase 3: Heatmap view (click to select a path, shown in side panel)
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
                mat = materialize_path(tables=tables, path_nodes=nodes, path_edges=edges)
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

        # Default selection: preselect the heatmap cell with the MOST paths.
        if "nav_selected_heatmap_cell" not in st.session_state:
            if cells:
                tuple_counts = [int(c.tuple_count) for c in cells]
                bins = compute_bins(tuple_counts, num_bins=10)

                grouped: Dict[Tuple[int, int], List[HeatmapCell]] = {}
                for c in cells:
                    bidx = bin_index(int(c.tuple_count), bins)
                    grouped.setdefault((bidx, int(c.node_count)), []).append(c)

                best_group = max(
                    grouped.items(),
                    key=lambda kv: (len(kv[1]), max(cc.tuple_count for cc in kv[1])),
                )[1]

                best_cell = max(best_group, key=lambda cc: cc.tuple_count)
                best_path_key = str(best_cell.path_key)
                path_keys = [str(cc.path_key) for cc in best_group]
                st.session_state.nav_selected_heatmap_cell = {
                    "best_path_key": best_path_key,
                    "path_keys": path_keys,
                }
            else:
                st.session_state.nav_selected_heatmap_cell = {}

        left, right = st.columns([0.20, 0.80], gap="small")

        with left:
            sel0 = st.session_state.get("nav_selected_heatmap_cell", {}) or {}
            selected_best_path_key = (
                str(sel0.get("best_path_key"))
                if isinstance(sel0, dict) and sel0.get("best_path_key") is not None
                else None
            )
            picked = render_joinpath_heatmap(
                cells=cells,
                query_param_key="nav_heatmap_sel",
                title="",
                height_px=400,
                selected_best_path_key=selected_best_path_key,
            )
            # Placeholder for selection summary
            heatmap_summary_slot = st.empty()

            # Semantic Types Section - directly under heatmap
            st.markdown("---")
            st.markdown("**Semantic Types**")
            
            if tables:
                sem_table_name = st.selectbox(
                    "Table",
                    options=sorted(tables.keys()),
                    key="nav_semtypes_table_select",
                    label_visibility="collapsed",
                )
                
                sem_df = _semantic_types_rows(tables, sem_table_name)
                if not sem_df.empty:
                    st.markdown(_semantic_types_badge_table(sem_df), unsafe_allow_html=True)
                else:
                    st.caption("No columns found.")
            else:
                st.caption("No tables available.")

        if isinstance(picked, dict) and picked:
            ts = picked.get("ts")
            last_ts = st.session_state.get("nav_heatmap_last_ts")
            st.session_state.nav_selected_heatmap_cell = {k: v for k, v in picked.items() if k != "ts"}
            if ts is not None and ts != last_ts:
                st.session_state.nav_heatmap_last_ts = ts
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()

        sel = st.session_state.get("nav_selected_heatmap_cell", {}) or {}
        path_keys = sel.get("path_keys") if isinstance(sel, dict) else None
        if not isinstance(path_keys, list):
            path_keys = []
        path_keys = [str(k) for k in path_keys if str(k) in by_key]
        path_keys = sorted(path_keys, key=lambda k: rank_by_key.get(k, 10**9))

        # Render selection summary below the heatmap (left column).
        if path_keys:
            tuple_bin_lo = sel.get("tuple_bin_lo") if isinstance(sel, dict) else None
            tuple_bin_hi = sel.get("tuple_bin_hi") if isinstance(sel, dict) else None
            node_count = sel.get("node_count") if isinstance(sel, dict) else None
            meta_parts = []
            if isinstance(node_count, (int, float)) and int(node_count) > 0:
                meta_parts.append(f"Nodes: {int(node_count)}")
            if isinstance(tuple_bin_lo, (int, float)) and isinstance(tuple_bin_hi, (int, float)):
                meta_parts.append(f"Tuple bin: {int(tuple_bin_lo):,}–{int(tuple_bin_hi):,}")
            meta_parts.append(f"Paths: {len(path_keys):,}")
            heatmap_summary_slot.caption("Showing " + " • ".join(meta_parts))
        else:
            heatmap_summary_slot.empty()

        with right:
            # Remove vertical gaps between successive join-path iframes.
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
            if not path_keys:
                st.info("Click a heatmap cell to view join paths.")
            else:
                # Render selected paths with the compact join-path cards.
                with st.container(height=820, border=False):
                    for k in path_keys:
                        nodes, score, edges = by_key[k]
                        rank = int(rank_by_key.get(k, 0)) or 0
                        hops = len(nodes) - 1

                        try:
                            materialized = materialize_path(
                                tables=tables, path_nodes=nodes, path_edges=edges
                            )
                            num_rows = materialized.shape[0]
                            num_cols = materialized.shape[1]
                            attributes = list(materialized.columns)
                        except Exception:
                            num_rows = 0
                            num_cols = 0
                            attributes = []

                        attributes_list = (
                            "".join([f"<span>{a}</span>" for a in attributes]) if num_cols > 0 else ""
                        )
                        dot_source = path_to_dot(nodes, edges, tables=tables)
                        dot_json = json.dumps(dot_source)

                        stats_html = f'''
                        <div class="sd-meta-stack">
                            <div class="sd-meta-chip sd-meta-chip-path">
                                <div class="sd-meta-k">Path</div>
                                <div class="sd-meta-v">#{rank}</div>
                            </div>
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
                                    background: linear-gradient(135deg, #FFFBEB 0%, #FFF7ED 100%);
                                    border-radius: 9px;
                                    border: 1px solid #FED7AA;
                                    box-shadow: 0 2px 4px rgba(249, 115, 22, 0.08);
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
                                    background: linear-gradient(135deg, #F97316 0%, #C4B5FD 100%);
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
                                #graph {{ width: 100%; display: block; }}
                                #graph svg {{
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
                                        <div id="graph"></div>
                                    </div>
                                </div>
                            </div>
                            <script>
                                const dot = {dot_json};
                                const viz = new Viz();
                                viz.renderSVGElement(dot)
                                    .then(function(element) {{
                                        element.style.height = '34px';
                                        element.style.width = 'auto';
                                        element.style.maxWidth = 'none';
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

        st.divider()
        if st.button("➡️ Proceed to next stage", type="primary", use_container_width=True):
            goto_stage(2)