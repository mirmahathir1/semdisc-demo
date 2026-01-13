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
        st.info("Please preprocess your data using the controls in the header first.")
        return

    tables: Dict[str, pd.DataFrame] = st.session_state.tables
    # Use a fixed low threshold for the demo to ensure join paths are always found
    # (hyperparameters are displayed but don't affect the simulated results)
    nav_min_edge_score = 0.10  # Fixed for demo simulation
    nav_max_hops = int(st.session_state.nav_max_hops)
    nav_top_k_per_table = 10

    graph = build_join_graph(tables, min_edge_score=nav_min_edge_score)

    # Join paths section (no tabs)
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

        def _has_semantic_join(es: List[EdgeInfo]) -> bool:
            return any(getattr(e, "join_type", "equi") == "semantic" for e in (es or []))

        # Prefer showing some semantic-join paths in the demo UI (still score-sorted within group).
        paths = sorted(
            by_nodes.values(),
            key=lambda x: (_has_semantic_join(x[2]), x[1]),
            reverse=True,
        )

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

            def _cell_group_has_semantic(group_cells: List[HeatmapCell]) -> bool:
                for cc in group_cells:
                    entry = by_key.get(str(cc.path_key))
                    if entry is None:
                        continue
                    _, _, es = entry
                    if any(getattr(e, "join_type", "equi") == "semantic" for e in (es or [])):
                        return True
                return False

            best_group = max(
                grouped.items(),
                key=lambda kv: (len(kv[1]), max(cc.tuple_count for cc in kv[1])),
            )[1]

            # If possible, prefer a group that contains at least one semantic-join path
            # so the demo always showcases semantic joins by default.
            if not _cell_group_has_semantic(best_group):
                semantic_groups = [g for g in grouped.values() if _cell_group_has_semantic(g)]
                if semantic_groups:
                    best_group = max(
                        semantic_groups,
                        key=lambda g: (len(g), max(cc.tuple_count for cc in g)),
                    )

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
        st.markdown(
            """
<div style="margin: 2px 0 10px 0; padding: 8px 10px; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
  <span style="display:flex; align-items:center; gap:6px;">
    <span style="width:20px; height:0; border-top:3px solid #2563EB; display:inline-block;"></span>
    <span style="color:#475569; font-size:0.88rem;">Equi-join</span>
  </span>
  <span style="display:flex; align-items:center; gap:6px;">
    <span style="width:20px; height:0; border-top:3px solid #F97316; display:inline-block;"></span>
    <span style="color:#475569; font-size:0.88rem;">Semantic join</span>
  </span>
</div>
            """,
            unsafe_allow_html=True,
        )
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
            height_px=200,
            selected_best_path_key=selected_best_path_key,
        )
        # Semantic Types Section - directly under heatmap
        if tables:
            sem_heading_col, sem_dropdown_col = st.columns([1, 1])
            with sem_heading_col:
                st.markdown("**Semantic Types**")
            with sem_dropdown_col:
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
            st.markdown("**Semantic Types**")
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


    with right:
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

                    # Show first 5 attributes, with "Show all" button if more
                    max_visible = 5
                    if num_cols > max_visible:
                        visible_attrs = "".join([f"<span class='attr-badge'>{a}</span>" for a in attributes[:max_visible]])
                        hidden_attrs = "".join([f"<span class='attr-badge'>{a}</span>" for a in attributes[max_visible:]])
                        attributes_list = f'{visible_attrs}<span class="hidden-attrs" style="display:none;">{hidden_attrs}</span><span class="attr-badge show-all-btn" role="button" tabindex="0" onclick="this.previousElementSibling.style.display=\'inline\';this.style.display=\'none\';">+{num_cols - max_visible} more</span>'
                    else:
                        attributes_list = "".join([f"<span class='attr-badge'>{a}</span>" for a in attributes]) if num_cols > 0 else ""
                    dot_source = path_to_dot(nodes, edges, tables=tables)
                    dot_json = json.dumps(dot_source)

                    stats_html = f'''
                    <div class="sd-meta-stack">
                        <div class="sd-meta-chip sd-meta-chip-plain">
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
                    graph_area_height = 66
                    meta_rail_height = 108
                    main_col_height = attrs_block_height + 6 + graph_area_height
                    content_h = max(meta_rail_height, main_col_height)
                    dynamic_height = int(content_h - 10)
                    dynamic_height = max(80, dynamic_height)

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
                                    padding: 8px 10px;
                                    background: linear-gradient(135deg, #FFFBEB 0%, #FFF7ED 100%);
                                    border-radius: 9px;
                                    border: 1px solid #FED7AA;
                                    box-shadow: 0 2px 4px rgba(249, 115, 22, 0.08);
                                    display: grid;
                                    grid-template-columns: 6.25% 93.75%;
                                    gap: 6px;
                                    min-height: fit-content;
                                    height: auto;
                                    margin: 0 0 2px 0;
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
                                .sd-meta-chip-plain {{
                                    background: transparent;
                                    border: none;
                                    padding: 2px 0;
                                }}
                                .sd-meta-chip-plain .sd-meta-v {{
                                    color: #6B7280;
                                    font-size: 0.70rem;
                                    font-weight: 600;
                                }}
                                .sd-meta-k {{
                                    font-size: 0.65rem;
                                    font-weight: 700;
                                    color: #6B7280;
                                    text-transform: uppercase;
                                    letter-spacing: 0.05em;
                                    line-height: 1.05;
                                }}
                                .sd-meta-v {{
                                    font-size: 0.85rem;
                                    font-weight: 800;
                                    color: #111827;
                                    line-height: 1.05;
                                }}
                                .sd-unified-attributes {{
                                    margin-top: 0px;
                                    flex: 1;
                                    min-height: fit-content;
                                    display: flex;
                                    flex-wrap: wrap;
                                    align-items: center;
                                    gap: 4px;
                                }}
                                .sd-unified-attributes-label {{
                                    font-size: 0.85rem;
                                    color: #6B7280;
                                    font-weight: 400;
                                    display: inline;
                                    margin-right: 4px;
                                }}
                                .sd-unified-attributes .attr-badge {{
                                    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
                                    font-size: 0.75rem;
                                    background: linear-gradient(135deg, #F5F3FF 0%, #DDD6FE 100%);
                                    padding: 3px 8px;
                                    border-radius: 6px;
                                    border: 1px solid #C4B5FD;
                                    color: #5B21B6;
                                    font-weight: 600;
                                }}
                                .show-all-btn {{
                                    cursor: pointer;
                                    user-select: none;
                                }}
                                .sd-unified-right {{
                                    display: block;
                                    background: #FFFFFF;
                                    border-radius: 8px;
                                    padding: 4px 4px 12px 4px;
                                    border: 1px solid #E2E8F0;
                                    height: 66px;
                                    overflow-x: auto;
                                    overflow-y: hidden;
                                }}
                                #graph {{ width: 100%; display: block; }}
                                #graph svg {{
                                    height: 44px !important;
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
                                        <span class="sd-unified-attributes-label">Attrs ({num_cols}):</span>{attributes_list}
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
                                        element.style.height = '44px';
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
