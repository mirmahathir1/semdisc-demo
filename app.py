from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

from utils import bundled_query, render_brand_header

import page_upload_preprocess
import page_navigate
import page_query_input
import page_retrieved_paths


def main() -> None:
    st.set_page_config(page_title="SemDisc", layout="wide")
    
    # Reduce top padding and define button colors
    st.markdown("""
        <style>
            /* Hide Streamlit's sidebar entirely (use in-page navigation instead). */
            section[data-testid="stSidebar"] { display: none !important; }
            div[data-testid="collapsedControl"] { display: none !important; }

            .main .block-container {
                padding-top: 1rem;
                padding-bottom: 1rem;
            }
            header[data-testid="stHeader"] {
                padding-top: 0.5rem;
                padding-bottom: 0.5rem;
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
            /* Secondary buttons - orange (default for non-primary buttons) */
            div[data-testid="stButton"]:not(.btn-reset):not(.btn-action):not(.btn-nav):not(.btn-download) > button:not(:disabled),
            div[data-testid="baseButton-secondary"]:not(.btn-reset):not(.btn-action):not(.btn-nav):not(.btn-download) > button:not(:disabled) {
                background-color: #F97316 !important;
                color: white !important;
                border-color: #F97316 !important;
            }
            div[data-testid="stButton"]:not(.btn-reset):not(.btn-action):not(.btn-nav):not(.btn-download) > button:not(:disabled):hover,
            div[data-testid="baseButton-secondary"]:not(.btn-reset):not(.btn-action):not(.btn-nav):not(.btn-download) > button:not(:disabled):hover {
                background-color: #EA580C !important;
                border-color: #EA580C !important;
            }
            /* Reset buttons - light orange */
            .btn-reset > button:not(:disabled),
            div[data-testid="stButton"].btn-reset > button:not(:disabled) {
                background-color: #FDBA74 !important;
                color: white !important;
                border-color: #FDBA74 !important;
            }
            .btn-reset > button:not(:disabled):hover,
            div[data-testid="stButton"].btn-reset > button:not(:disabled):hover {
                background-color: #FB923C !important;
                border-color: #FB923C !important;
            }
            /* Action buttons (Add, etc.) - light orange (same as download) */
            .btn-action > button:not(:disabled),
            div[data-testid="stButton"].btn-action > button:not(:disabled) {
                background-color: #FDBA74 !important;
                color: white !important;
                border-color: #FDBA74 !important;
            }
            .btn-action > button:not(:disabled):hover,
            div[data-testid="stButton"].btn-action > button:not(:disabled):hover {
                background-color: #FB923C !important;
                border-color: #FB923C !important;
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
            /* Download buttons - light orange */
            .btn-download > button:not(:disabled),
            div[data-testid="stDownloadButton"].btn-download > button:not(:disabled),
            div[data-testid="stButton"].btn-download > button:not(:disabled) {
                background-color: #FDBA74 !important;
                color: white !important;
                border-color: #FDBA74 !important;
            }
            .btn-download > button:not(:disabled):hover,
            div[data-testid="stDownloadButton"].btn-download > button:not(:disabled):hover,
            div[data-testid="stButton"].btn-download > button:not(:disabled):hover {
                background-color: #FB923C !important;
                border-color: #FB923C !important;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Inject JavaScript via components.html to classify buttons by text content
    # This runs in an iframe but accesses the parent document
    components.html("""
        <script>
            function addButtonClasses() {
                const parent = window.parent.document;
                
                // Target all button containers in the parent document
                const buttonContainers = parent.querySelectorAll('div[data-testid="stButton"], div[data-testid="baseButton-secondary"], div[data-testid="stDownloadButton"]');
                
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
    """, height=0)
    
    render_brand_header()

    # Session state initialization
    st.session_state.setdefault("processed_ready", False)
    st.session_state.setdefault("datalake_token", None)
    st.session_state.setdefault("tables", {})
    st.session_state.setdefault("query_ready", False)
    st.session_state.setdefault("query_df", None)
    st.session_state.setdefault("query_submitted", False)
    # Hyperparameters (mirrors `dice/stages/offline_pipeline_cli.py` CLI flags)
    st.session_state.setdefault("hp_simhash_size", 128)
    st.session_state.setdefault("hp_num_hashes_minhash", 128)
    st.session_state.setdefault("hp_join_edge_threshold", 0.10)
    st.session_state.setdefault("hp_semantic_type_similarity_threshold", 0.70)
    # Default to longer paths so the demo naturally shows 4–5 node join paths.
    st.session_state.setdefault("hp_simple_path_max_length", 5)
    # Defaults for downstream screens (seeded from preprocessing hyperparameters)
    st.session_state.setdefault("nav_min_edge_score", float(st.session_state.hp_join_edge_threshold))
    st.session_state.setdefault("nav_max_hops", int(st.session_state.hp_simple_path_max_length))
    st.session_state.setdefault("ret_min_edge_score", float(st.session_state.hp_join_edge_threshold))
    st.session_state.setdefault("ret_max_hops", int(st.session_state.hp_simple_path_max_length))
    st.session_state.setdefault("ret_top_k", 10)
    # Phase 4: Query table selection state (used on Retrieved Join Paths page)
    st.session_state.setdefault("selected_row", None)
    st.session_state.setdefault("selected_column", None)
    st.session_state.setdefault("page_idx", 0)
    st.session_state.setdefault("preprocess_log", [])

    # Always keep a prepopulated query table available by default.
    if st.session_state.query_df is None:
        st.session_state.query_df = bundled_query()
        st.session_state.query_ready = False
        st.session_state.query_submitted = False

    page_idx = int(st.session_state.page_idx)
    page_idx = max(0, min(3, page_idx))
    st.session_state.page_idx = page_idx

    # Route to the appropriate page
    if page_idx == 0:
        page_upload_preprocess.render()
    elif page_idx == 1:
        page_navigate.render()
    elif page_idx == 2:
        page_query_input.render()
    else:
        page_retrieved_paths.render()


if __name__ == "__main__":
    main()
