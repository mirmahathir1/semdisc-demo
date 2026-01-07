from __future__ import annotations

import json
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from utils import (
    EdgeInfo,
    build_join_graph,
    goto_stage,
    materialize_path,
    path_to_dot,
    simple_paths_topk,
)


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

    st.subheader("Join paths")
    if not graph:
        st.warning("No join edges found with the current settings.")
    elif not paths:
        st.warning("No join paths found with the current settings.")
    else:
        # CSS for styling path cards
        st.markdown(
            """
<style>
  .sd-pathcard { 
    border: 1px solid #FED7AA; 
    background: linear-gradient(135deg, #FFFBEB 0%, #FFF7ED 100%); 
    border-radius: 12px; 
    padding: 16px 18px; 
    margin-bottom: 14px; 
    box-shadow: 0 2px 4px rgba(249, 115, 22, 0.06);
    transition: all 0.2s ease;
  }
  .sd-pathcard:hover {
    box-shadow: 0 4px 8px rgba(249, 115, 22, 0.12);
    border-color: #FDBA74;
  }
  .sd-pathcard-wrapper {
    border: 2px solid #FED7AA !important; 
    background: linear-gradient(135deg, #FFFBEB 0%, #FFF7ED 100%) !important; 
    border-radius: 16px !important; 
    padding: 20px 24px !important; 
    margin-bottom: 16px !important; 
    box-shadow: 0 4px 6px rgba(249, 115, 22, 0.08), 0 1px 3px rgba(249, 115, 22, 0.1) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
  }
  .sd-pathcard-wrapper::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, #FDBA74 0%, #C4B5FD 50%, #FDBA74 100%);
    background-size: 200% 100%;
    animation: shimmer 3s ease-in-out infinite;
  }
  @keyframes shimmer {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
  }
  .sd-pathcard-wrapper:hover {
    box-shadow: 0 10px 15px rgba(0,0,0,0.1), 0 4px 6px rgba(0,0,0,0.05) !important;
    border-color: #FDBA74 !important;
    transform: translateY(-2px);
  }
  /* Make following elements appear as part of the card */
  .sd-pathcard-wrapper ~ * {
    margin-top: 0 !important;
  }
  /* Target the container that holds columns */
  div[data-card-start="true"] {
    position: relative;
  }
  /* Style elements that come after the card start marker */
  div[data-card-start="true"] ~ div {
    margin-left: 0 !important;
    margin-right: 0 !important;
  }
  /* JavaScript will wrap columns in the card */
  .sd-pathhead { 
    display: flex; 
    justify-content: flex-start; 
    gap: 10px; 
    align-items: center;
    margin-bottom: 4px;
    padding-bottom: 12px;
    border-bottom: 2px solid #FED7AA;
  }
  .sd-rank { 
    font-weight: 700; 
    color: #0F172A; 
    font-size: 1.15rem;
    letter-spacing: -0.02em;
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }
  .sd-rank-number {
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
    text-align: center;
  }
  .sd-rank-path {
    color: #4B5563;
    font-weight: 600;
  }
  .sd-rank-label {
    color: #1E293B;
    font-weight: 700;
    font-size: 1.1rem;
    letter-spacing: -0.01em;
    text-transform: uppercase;
    font-size: 0.85rem;
    letter-spacing: 0.05em;
  }
  .sd-rank-hops {
    color: #5B21B6;
    font-weight: 600;
    font-size: 0.95em;
    padding: 4px 10px;
    background: linear-gradient(135deg, #F5F3FF 0%, #DDD6FE 100%);
    border-radius: 6px;
    border: 1px solid #C4B5FD;
  }
  .sd-meta { 
    font-size: 0.88rem; 
    color: #374151; 
    margin-top: 10px;
    padding: 8px 12px;
    background: #FFF7ED;
    border-radius: 8px;
    border-left: 3px solid #F97316;
  }
  .sd-meta b {
    color: #4B5563;
    font-weight: 650;
  }
  .sd-meta-keys {
    color: #1F2937;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    font-weight: 500;
    margin-left: 4px;
  }
  .sd-path { 
    margin-top: 10px; 
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; 
    font-size: 0.9rem; 
    color: #111827; 
    background: linear-gradient(135deg, #F5F3FF 0%, #EDE9FE 100%); 
    border: 1px solid #C4B5FD; 
    border-radius: 10px; 
    padding: 10px 14px;
    font-weight: 500;
    letter-spacing: 0.02em;
  }
  .sd-unified-component {
    padding: 20px 24px;
    background: linear-gradient(135deg, #FFFBEB 0%, #FFF7ED 100%);
    border-radius: 12px;
    border: 2px solid #FED7AA;
    box-shadow: 0 2px 4px rgba(249, 115, 22, 0.08);
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
  }
  .sd-unified-left {
    display: flex;
    flex-direction: column;
  }
  .sd-unified-right {
    display: flex;
    flex-direction: column;
    align-items: stretch;
  }
  .sd-unified-header {
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 16px;
    padding-bottom: 16px;
    border-bottom: 2px solid #FED7AA;
  }
  .sd-unified-title {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
  }
  .sd-unified-label {
    color: #1E293B;
    font-weight: 700;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .sd-unified-rank {
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
  }
  .sd-unified-hops {
    display: inline-flex;
    align-items: center;
    padding: 6px 14px;
    background: linear-gradient(135deg, #A78BFA 0%, #C4B5FD 100%);
    color: white;
    border-radius: 8px;
    font-weight: 700;
    font-size: 0.9rem;
    box-shadow: 0 2px 4px rgba(167, 139, 250, 0.3);
  }
  .sd-unified-stats {
    display: flex;
    align-items: flex-start;
    gap: 24px;
    margin-left: auto;
  }
  .sd-stat-item {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 4px;
  }
  .sd-stat-label {
    font-size: 0.85rem;
    color: #475569;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .sd-stat-value {
    font-size: 1.1rem;
    color: #334155;
    font-weight: 700;
  }
  .sd-unified-attributes {
    margin-top: 12px;
  }
  .sd-unified-attributes-label {
    font-size: 0.95rem;
    color: #334155;
    font-weight: 700;
    margin-bottom: 10px;
    display: block;
  }
  .sd-unified-attributes-list {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    font-size: 0.88rem;
    color: #1E293B;
    font-weight: 500;
    line-height: 1.8;
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }
  .sd-unified-attributes-list span {
    background: linear-gradient(135deg, #F5F3FF 0%, #DDD6FE 100%);
    padding: 4px 10px;
    border-radius: 6px;
    border: 1px solid #C4B5FD;
    color: #5B21B6;
    font-weight: 600;
  }
</style>
<script>
(function() {
    function wrapCards() {
        const cardStarts = document.querySelectorAll('[data-card-start="true"]');
        cardStarts.forEach(cardStart => {
            if (cardStart.dataset.processed === 'true') return;
            const cardId = cardStart.id;
            const cardEnd = document.querySelector(`[data-card-end="true"][data-card-id="${cardId}"]`);
            if (!cardEnd) return;
            
            const parent = cardStart.parentElement;
            if (!parent) return;
            
            // Get all siblings between start and end
            const allSiblings = Array.from(parent.children);
            const startIndex = allSiblings.indexOf(cardStart);
            const endIndex = allSiblings.indexOf(cardEnd);
            
            if (startIndex === -1 || endIndex === -1 || startIndex >= endIndex) return;
            
            // Collect ALL elements between start and end (be more aggressive)
            const elementsToMove = [];
            for (let i = startIndex + 1; i < endIndex; i++) {
                const sibling = allSiblings[i];
                if (sibling && sibling.nodeType === 1 && sibling !== cardEnd) {
                    // Move any element node (be more inclusive)
                    if (!cardStart.contains(sibling)) {
                        elementsToMove.push(sibling);
                    }
                }
            }
            
            // Move all elements into the card wrapper
            elementsToMove.forEach(el => {
                if (el && el.parentNode) {
                    try {
                        cardStart.appendChild(el);
                    } catch (e) {
                        // Element might already be moved
                    }
                }
            });
            
            // Also look for Streamlit column containers anywhere in the document
            // that appear after the card start
            const allColumnContainers = document.querySelectorAll('[data-testid*="column"]');
            allColumnContainers.forEach(col => {
                // Check if this column comes after cardStart in document order
                const colRect = col.getBoundingClientRect();
                const startRect = cardStart.getBoundingClientRect();
                const endRect = cardEnd.getBoundingClientRect();
                
                if (colRect.top >= startRect.top && 
                    colRect.top <= endRect.bottom && 
                    !cardStart.contains(col) &&
                    col.parentNode) {
                    try {
                        cardStart.appendChild(col);
                    } catch (e) {
                        // Already moved or can't move
                    }
                }
            });
            
            // Also look for any divs with Streamlit content between start and end
            let current = cardStart.nextElementSibling;
            while (current && current !== cardEnd) {
                if (current.nodeType === 1 && 
                    !cardStart.contains(current) &&
                    (current.hasAttribute('data-testid') || 
                     current.querySelector('[data-testid]') ||
                     current.textContent.trim().length > 0)) {
                    try {
                        cardStart.appendChild(current);
                        // Reset current since DOM changed
                        current = cardStart.nextElementSibling;
                    } catch (e) {
                        current = current.nextElementSibling;
                    }
                } else {
                    current = current.nextElementSibling;
                }
            }
            
            // Remove the end marker
            if (cardEnd && cardEnd.parentNode) {
                cardEnd.parentNode.removeChild(cardEnd);
            }
            
            // Move graphviz chart into the right column
            const graphvizContainer = cardStart.querySelector(`[id^="graphviz-container-"]`);
            if (graphvizContainer) {
                // Find graphviz elements that come after the card start but before the end
                const parent = cardStart.parentElement;
                if (parent) {
                    const allSiblings = Array.from(parent.children);
                    const startIndex = allSiblings.indexOf(cardStart);
                    const endIndex = allSiblings.indexOf(cardEnd);
                    
                    // Look for graphviz in siblings between start and end
                    for (let i = startIndex + 1; i < endIndex; i++) {
                        const sibling = allSiblings[i];
                        if (!sibling || cardStart.contains(sibling)) continue;
                        
                        // Check if sibling contains graphviz
                        const graphvizEl = sibling.querySelector('[data-testid*="graphviz"]');
                        if (graphvizEl) {
                            // Find the top-level container for this graphviz
                            let container = graphvizEl.closest('div[data-testid]');
                            if (container && !cardStart.contains(container)) {
                                if (graphvizContainer.children.length === 0) {
                                    graphvizContainer.appendChild(container);
                                }
                                break;
                            }
                        }
                        // Also check if sibling itself is a graphviz container
                        if (sibling.hasAttribute('data-testid') && 
                            sibling.getAttribute('data-testid').includes('graphviz')) {
                            if (graphvizContainer.children.length === 0) {
                                graphvizContainer.appendChild(sibling);
                            }
                            break;
                        }
                    }
                }
                
                // Also search more broadly for graphviz near this card
                if (graphvizContainer.children.length === 0) {
                    const cardRect = cardStart.getBoundingClientRect();
                    const endRect = cardEnd.getBoundingClientRect();
                    
                    // Find all graphviz elements in the document
                    const allGraphviz = document.querySelectorAll('[data-testid*="graphviz"]');
                    allGraphviz.forEach(gv => {
                        const gvRect = gv.getBoundingClientRect();
                        // Check if graphviz is visually between card start and end
                        if (gvRect.top >= cardRect.top && 
                            gvRect.top <= endRect.bottom &&
                            !cardStart.contains(gv)) {
                            const container = gv.closest('div[data-testid]');
                            if (container && graphvizContainer.children.length === 0) {
                                graphvizContainer.appendChild(container);
                            }
                        }
                    });
                }
            }
            
            // Mark as processed
            cardStart.dataset.processed = 'true';
        });
    }
    
    // Function to move graphviz charts into their containers
    function moveGraphvizCharts() {
        const containers = document.querySelectorAll('[id^="graphviz-container-"]');
        containers.forEach(container => {
            if (container.children.length > 0) return; // Already has content
            
            const containerId = container.id;
            const cardIndex = containerId.match(/\d+/);
            if (!cardIndex) return;
            
            const cardStart = document.querySelector(`[id="pathcard-${cardIndex[0]}"]`);
            if (!cardStart) return;
            
            const cardEnd = document.querySelector(`[data-card-end="true"][data-card-id="pathcard-${cardIndex[0]}"]`);
            
            // Get the card's bounding box for visual matching
            const cardRect = cardStart.getBoundingClientRect();
            const endRect = cardEnd ? cardEnd.getBoundingClientRect() : null;
            
            // Search more broadly - look for graphviz near this card
            const allGraphviz = document.querySelectorAll('[data-testid*="graphviz"]');
            let foundGraphviz = null;
            
            allGraphviz.forEach(gv => {
                if (foundGraphviz) return; // Already found one
                
                const gvRect = gv.getBoundingClientRect();
                // Check if graphviz is visually positioned near this card
                // It should be below the card start and not too far away
                if (gvRect.top >= cardRect.top && 
                    gvRect.top <= (endRect ? endRect.bottom : cardRect.bottom + 500) &&
                    Math.abs(gvRect.left - cardRect.left) < 1000 && // Within reasonable horizontal distance
                    !cardStart.contains(gv)) {
                    
                    // Find the top-level container for this graphviz
                    let graphvizContainer = gv.closest('div[data-testid]');
                    if (!graphvizContainer) {
                        graphvizContainer = gv.parentElement;
                    }
                    
                    // Make sure it's not already inside a card
                    if (graphvizContainer && 
                        !graphvizContainer.closest('[id^="pathcard-"]') &&
                        !container.contains(graphvizContainer)) {
                        foundGraphviz = graphvizContainer;
                    }
                }
            });
            
            // If found, move it to the container
            if (foundGraphviz && container.children.length === 0) {
                container.appendChild(foundGraphviz);
            }
            
            // Also check siblings in the parent
            if (!foundGraphviz) {
                const parent = cardStart.parentElement;
                if (parent) {
                    const allSiblings = Array.from(parent.children);
                    const startIndex = allSiblings.indexOf(cardStart);
                    const endIndex = cardEnd ? allSiblings.indexOf(cardEnd) : allSiblings.length;
                    
                    // Search from start to end, and a bit beyond
                    for (let i = startIndex + 1; i < Math.min(endIndex + 3, allSiblings.length); i++) {
                        const sibling = allSiblings[i];
                        if (!sibling || cardStart.contains(sibling)) continue;
                        
                        // Check for graphviz
                        const graphvizEl = sibling.querySelector('[data-testid*="graphviz"]');
                        if (graphvizEl) {
                            const graphvizContainer = graphvizEl.closest('div[data-testid]');
                            if (graphvizContainer && !cardStart.contains(graphvizContainer) && container.children.length === 0) {
                                container.appendChild(graphvizContainer);
                                return;
                            }
                        }
                        // Check if sibling itself is graphviz
                        if (sibling.hasAttribute('data-testid') && 
                            sibling.getAttribute('data-testid').includes('graphviz') &&
                            container.children.length === 0) {
                            container.appendChild(sibling);
                            return;
                        }
                    }
                }
            }
        });
    }
    
    // Run wrapper function
    function runWrapper() {
        wrapCards();
        moveGraphvizCharts();
    }
    
    // Run after DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', runWrapper);
    } else {
        runWrapper();
    }
    
    // Use MutationObserver to catch dynamically added content
    const observer = new MutationObserver(function(mutations) {
        let shouldRun = false;
        mutations.forEach(function(mutation) {
            if (mutation.addedNodes.length > 0) {
                shouldRun = true;
            }
        });
        if (shouldRun) {
            setTimeout(() => {
                wrapCards();
                moveGraphvizCharts();
            }, 50);
        }
    });
    
    if (document.body) {
        observer.observe(document.body, { childList: true, subtree: true });
    }
    
    // Also run after delays to catch graphviz charts
    setTimeout(() => { wrapCards(); moveGraphvizCharts(); }, 100);
    setTimeout(() => { wrapCards(); moveGraphvizCharts(); }, 300);
    setTimeout(() => { wrapCards(); moveGraphvizCharts(); }, 500);
    setTimeout(() => { wrapCards(); moveGraphvizCharts(); }, 1000);
    setTimeout(() => { wrapCards(); moveGraphvizCharts(); }, 2000);
    setTimeout(() => { wrapCards(); moveGraphvizCharts(); }, 3000);
})();
</script>
""",
            unsafe_allow_html=True,
        )
        
        # Use Streamlit's native scrollable container (height parameter)
        with st.container(height=600, border=True):
            for idx, (nodes, score, edges) in enumerate(paths):
                if len(nodes) <= 1:
                    continue
                rank = idx + 1
                hops = len(nodes) - 1
                path_txt = " -- ".join(nodes)
                
                # Materialize path to get metadata
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
                
                # Create the card wrapper - include opening div
                card_id = f"pathcard-{idx}"
                
                # Build attributes list
                attributes_list = ''.join([f'<span>{attr}</span>' for attr in attributes]) if num_cols > 0 else ''
                
                # Generate the DOT source for the graph
                dot_source = path_to_dot(nodes, edges, tables=tables)
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
                # Base height: header (~70px) + padding (~50px) + label (~30px) + margin (~20px) = ~170px
                # Each row of attributes adds ~38px (badge height + gap)
                # Assume ~3 attributes per row (with padding/borders, names can be long)
                # Also ensure minimum height for the graph visualization (~150px)
                attr_rows = max(1, (num_cols + 2) // 3)  # Ceiling division
                content_height = 170 + (attr_rows * 38)
                graph_min_height = 180  # Minimum for graph to display properly
                dynamic_height = max(content_height, graph_min_height) + 30  # Extra buffer
                
                # Create the entire card HTML including graphviz in a single component
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
                        <div class="sd-unified-right" id="graphviz-container-{idx}">
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

    st.divider()
    if st.button("➡️ Proceed to next stage", type="primary", use_container_width=True):
        goto_stage(2)

