from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

from utils import (
    bundled_datalake,
    bundled_query,
    ensure_nonempty_tables,
    goto_stage,
    render_brand_header,
    simulate_preprocessing,
)

import page_navigate
import page_query_input
import page_retrieved_paths


def main() -> None:
    st.set_page_config(page_title="SemDisc", layout="wide")
    
    # Reduce top padding and define button colors
    st.markdown("""
        <style>
            /* Streamlit often reserves a tall top header/toolbar area even when unused.
               Collapse it so there's no huge whitespace above our brand header. */
            header[data-testid="stHeader"] {
                height: 0px !important;
                min-height: 0px !important;
                padding: 0px !important;
                margin: 0px !important;
                border: 0px !important;
            }
            header[data-testid="stHeader"] * { display: none !important; }
            div[data-testid="stToolbar"] { display: none !important; }
            div[data-testid="stDecoration"] { display: none !important; }

            /* Hide Streamlit's sidebar entirely (use in-page navigation instead). */
            section[data-testid="stSidebar"] { display: none !important; }
            div[data-testid="collapsedControl"] { display: none !important; }

            /* Some Streamlit versions add a top offset/padding on the app container to
               make room for the (fixed) header. Remove that too. */
            div[data-testid="stAppViewContainer"],
            div[data-testid="stAppViewContainer"] > .main,
            div[data-testid="stAppViewContainer"] > .main > div,
            section.main,
            section.main > div {
                padding-top: 0px !important;
                margin-top: 0px !important;
            }

            /* Make sure the main content starts near the top. */
            /* Streamlit has changed container class/testids across versions; cover both. */
            div[data-testid="stAppViewBlockContainer"],
            div[data-testid="stMainBlockContainer"],
            div.block-container,
            div[data-testid="stAppViewContainer"] .main .block-container,
            .main .block-container {
                padding-top: 0.25rem !important;
                padding-bottom: 1rem !important;
            }
            /* Primary buttons - keep default Streamlit blue (targeted by Streamlit's baseButton-primary) */
            div[data-testid="baseButton-primary"] > button:not(:disabled),
            div[data-testid="baseButton-primary"] > button:not(:disabled):hover {
                background-color: rgb(19, 102, 255) !important;
                color: white !important;
                border-color: rgb(19, 102, 255) !important;
            }
            div[data-testid="baseButton-primary"] > button:not(:disabled):hover {
                background-color: rgb(15, 82, 204) !important;
                border-color: rgb(15, 82, 204) !important;
            }
            /* Secondary buttons - blue (default for non-primary buttons) */
            div[data-testid="stButton"]:not(.btn-reset):not(.btn-action):not(.btn-nav):not(.btn-download) > button:not(:disabled),
            div[data-testid="baseButton-secondary"]:not(.btn-reset):not(.btn-action):not(.btn-nav):not(.btn-download) > button:not(:disabled) {
                background-color: #2563EB !important;
                color: white !important;
                border-color: #2563EB !important;
            }
            div[data-testid="stButton"]:not(.btn-reset):not(.btn-action):not(.btn-nav):not(.btn-download) > button:not(:disabled):hover,
            div[data-testid="baseButton-secondary"]:not(.btn-reset):not(.btn-action):not(.btn-nav):not(.btn-download) > button:not(:disabled):hover {
                background-color: #1D4ED8 !important;
                border-color: #1D4ED8 !important;
            }
            /* Reset buttons - light blue */
            .btn-reset > button:not(:disabled),
            div[data-testid="stButton"].btn-reset > button:not(:disabled) {
                background-color: #93C5FD !important;
                color: white !important;
                border-color: #93C5FD !important;
            }
            .btn-reset > button:not(:disabled):hover,
            div[data-testid="stButton"].btn-reset > button:not(:disabled):hover {
                background-color: #60A5FA !important;
                border-color: #60A5FA !important;
            }
            /* Action buttons (Add, etc.) - light blue (same as download) */
            .btn-action > button:not(:disabled),
            div[data-testid="stButton"].btn-action > button:not(:disabled) {
                background-color: #93C5FD !important;
                color: white !important;
                border-color: #93C5FD !important;
            }
            .btn-action > button:not(:disabled):hover,
            div[data-testid="stButton"].btn-action > button:not(:disabled):hover {
                background-color: #60A5FA !important;
                border-color: #60A5FA !important;
            }
            /* Navigation buttons (Back) - gray */
            .btn-nav > button:not(:disabled),
            div[data-testid="stButton"].btn-nav > button:not(:disabled) {
                background-color: #6B7280 !important;
                color: white !important;
                border-color: #6B7280 !important;
            }
            .btn-nav > button:not(:disabled):hover,
            div[data-testid="stButton"].btn-nav > button:not(:disabled):hover {
                background-color: #4B5563 !important;
                border-color: #4B5563 !important;
            }
            /* Download buttons - light blue */
            .btn-download > button:not(:disabled),
            div[data-testid="stDownloadButton"].btn-download > button:not(:disabled),
            div[data-testid="stButton"].btn-download > button:not(:disabled) {
                background-color: #93C5FD !important;
                color: white !important;
                border-color: #93C5FD !important;
            }
            .btn-download > button:not(:disabled):hover,
            div[data-testid="stDownloadButton"].btn-download > button:not(:disabled):hover,
            div[data-testid="stButton"].btn-download > button:not(:disabled):hover {
                background-color: #60A5FA !important;
                border-color: #60A5FA !important;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Session state initialization (must happen before render_brand_header)
    st.session_state.setdefault("processed_ready", False)
    st.session_state.setdefault("datalake_token", None)
    st.session_state.setdefault("tables", {})
    st.session_state.setdefault("query_ready", False)
    st.session_state.setdefault("query_df", None)
    st.session_state.setdefault("query_submitted", False)
    # Hyperparameters (mirrors `dice/stages/offline_pipeline_cli.py` CLI flags)
    st.session_state.setdefault("hp_simhash_size", 32)
    st.session_state.setdefault("hp_num_hashes_minhash", 128)
    st.session_state.setdefault("hp_join_edge_threshold", 0.80)
    st.session_state.setdefault("hp_semantic_type_similarity_threshold", 0.70)
    # Default to longer paths so the demo naturally shows 4–5 node join paths.
    st.session_state.setdefault("hp_simple_path_max_length", 6)
    # Defaults for downstream screens (seeded from preprocessing hyperparameters)
    st.session_state.setdefault("nav_min_edge_score", float(st.session_state.hp_join_edge_threshold))
    st.session_state.setdefault("nav_max_hops", int(st.session_state.hp_simple_path_max_length))
    st.session_state.setdefault("ret_min_edge_score", float(st.session_state.hp_join_edge_threshold))
    st.session_state.setdefault("ret_max_hops", int(st.session_state.hp_simple_path_max_length))
    st.session_state.setdefault("ret_top_k", 10)
    # Phase 4: Query table selection state (used on Retrieved Join Paths page)
    st.session_state.setdefault("selected_row", None)
    st.session_state.setdefault("selected_column", None)
    st.session_state.setdefault("page_idx", 1)
    st.session_state.setdefault("preprocess_log", [])
    
    # Always keep a prepopulated query table available by default.
    if st.session_state.query_df is None:
        st.session_state.query_df = bundled_query()
        st.session_state.query_ready = False
        st.session_state.query_submitted = False
    
    render_brand_header()
    
    # Handle preprocessing from header controls
    preprocess_clicked = st.session_state.get("header_preprocess_clicked", False)
    
    if preprocess_clicked:
        # Keep ordering aligned to `offline_pipeline_cli.py` routines (0..12),
        # but only simulate the main milestones.
        routine_steps = {
            0: "Deleting processed data",
            1: "Copying data lake",
            2: "Sanitizing tables",
            3: "Extracting semantic types",
            4: "Computing table embeddings",
            6: "Computing MinHash similarities",
            7: "Computing semantic hashes",
            8: "Computing diversity index",
            9: "Computing semantic type similarities",
            10: "Detecting numeric columns",
            11: "Building join graph",
            12: "Building join path index",
        }
        steps = [routine_steps[i] for i in range(0, 13) if i in routine_steps]
        # simulate_preprocessing handles the spinner/status display internally
        simulate_preprocessing(steps=steps, container=None)
        
        # Load bundled datalake (demo uses sample data)
        tables = bundled_datalake()
        tables = ensure_nonempty_tables(tables)
        if not tables:
            st.error("No tables found in the data lake.")
            st.session_state.header_preprocess_clicked = False
        else:
            st.session_state.tables = tables
            st.session_state.processed_ready = True
            # Seed downstream defaults from preprocessing hyperparameters.
            st.session_state.nav_min_edge_score = float(st.session_state.hp_join_edge_threshold)
            st.session_state.nav_max_hops = int(st.session_state.hp_simple_path_max_length)
            st.session_state.ret_min_edge_score = float(st.session_state.hp_join_edge_threshold)
            st.session_state.ret_max_hops = int(st.session_state.hp_simple_path_max_length)
            st.session_state.header_preprocess_clicked = False
    
    # Inject JavaScript via components.html to classify buttons by text content.
    # NOTE: `height=0` can still reserve vertical space in some Streamlit builds,
    # so keep this iframe at 1px and place it AFTER the brand header so it never
    # creates whitespace above "SemDisc".
    # This runs in an iframe but accesses the parent document.
    components.html(
        """
        <script>
            function addButtonClasses() {
                const parent = window.parent.document;

                // Target all button containers in the parent document
                const buttonContainers = parent.querySelectorAll(
                    'div[data-testid="stButton"], div[data-testid="baseButton-secondary"], div[data-testid="stDownloadButton"]'
                );

                buttonContainers.forEach(wrapper => {
                    const btn = wrapper.querySelector('button');
                    if (!btn) return;

                    const btnText = btn.textContent.trim();

                    // Skip primary buttons
                    if (wrapper.closest('div[data-testid="baseButton-primary"]')) {
                        return;
                    }

                    // Remove existing classes
                    wrapper.classList.remove('btn-reset', 'btn-action', 'btn-nav', 'btn-download');

                    // Classify buttons by text content
                    if (btnText.includes('Reset') || btnText.toLowerCase().includes('reset')) {
                        wrapper.classList.add('btn-reset');
                    } else if (btnText.includes('Add') || btnText.includes('➕')) {
                        wrapper.classList.add('btn-action');
                    } else if (btnText.includes('Back') || btnText.toLowerCase().includes('back')) {
                        wrapper.classList.add('btn-nav');
                    } else if (btnText.includes('Download') || btnText.toLowerCase().includes('download')) {
                        wrapper.classList.add('btn-download');
                    }
                });
            }

            // Run immediately and on intervals to catch dynamically added buttons
            addButtonClasses();
            setTimeout(addButtonClasses, 100);
            setTimeout(addButtonClasses, 300);
            setTimeout(addButtonClasses, 500);
            setTimeout(addButtonClasses, 1000);

            // Watch for DOM changes in parent document
            const observer = new MutationObserver(addButtonClasses);
            observer.observe(window.parent.document.body, { childList: true, subtree: true });
        </script>
        """,
        height=1,
    )

    page_idx = int(st.session_state.page_idx)
    # Skip page 0 (preprocessing page), start from page 1
    page_idx = max(1, min(3, page_idx))
    st.session_state.page_idx = page_idx

    # Route to the appropriate page (page 0 removed, preprocessing is in header)
    if page_idx == 1:
        page_navigate.render()
    elif page_idx == 2:
        page_query_input.render()
    else:
        page_retrieved_paths.render()


if __name__ == "__main__":
    main()
