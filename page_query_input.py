from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from utils import bundled_query, goto_stage


def render() -> None:
    """Render Stage 2: Input Query Table."""
    st.subheader("Input Query Table")
    
    # Add custom CSS and JavaScript for semantic type expander styling and scrollable table
    st.markdown("""
        <style>
            /* Style the semantic type expander with custom class */
            .semantic-type-expander {
                border: 2px solid #FDBA74 !important;
                border-radius: 12px !important;
                background: linear-gradient(135deg, #FFFBEB 0%, #FFF7ED 100%) !important;
                box-shadow: 0 4px 6px rgba(249, 115, 22, 0.1) !important;
                margin-bottom: 16px !important;
                overflow: hidden !important;
            }
            /* Target expander header button */
            .semantic-type-expander button[aria-expanded] {
                background: linear-gradient(135deg, #F97316 0%, #FDBA74 100%) !important;
                border-radius: 10px 10px 0 0 !important;
                padding: 12px 16px !important;
                color: white !important;
                font-weight: 600 !important;
                font-size: 1.05em !important;
                border: none !important;
                width: 100% !important;
            }
            /* Style expander header on hover */
            .semantic-type-expander button[aria-expanded]:hover {
                background: linear-gradient(135deg, #EA580C 0%, #F97316 100%) !important;
            }
            /* Style expander content area */
            .semantic-type-expander > div[data-testid="stExpanderDetails"] {
                padding: 20px !important;
                background: linear-gradient(135deg, #FFFFFF 0%, #FFFBEB 100%) !important;
                border-radius: 0 0 10px 10px !important;
            }
            /* Make query table scrollable with better styling */
            div[data-testid="stDataEditor"] {
                border: 1px solid #FED7AA !important;
                border-radius: 8px !important;
                box-shadow: 0 2px 4px rgba(249, 115, 22, 0.08) !important;
                overflow: hidden !important;
            }
            /* Ensure the data editor content is scrollable */
            div[data-testid="stDataEditor"] > div {
                max-height: 420px !important;
                overflow-y: auto !important;
                overflow-x: auto !important;
            }
            /* Style the scrollbar */
            div[data-testid="stDataEditor"] > div::-webkit-scrollbar {
                width: 8px;
                height: 8px;
            }
            div[data-testid="stDataEditor"] > div::-webkit-scrollbar-track {
                background: #FFF7ED;
                border-radius: 4px;
            }
            div[data-testid="stDataEditor"] > div::-webkit-scrollbar-thumb {
                background: #FDBA74;
                border-radius: 4px;
            }
            div[data-testid="stDataEditor"] > div::-webkit-scrollbar-thumb:hover {
                background: #F97316;
            }
        </style>
        <script>
            function styleSemanticTypeExpander() {
                const expanders = document.querySelectorAll('div[data-testid="stExpander"]');
                expanders.forEach(expander => {
                    const button = expander.querySelector('button[aria-expanded]');
                    if (button && button.textContent.includes('Edit Semantic Types')) {
                        expander.classList.add('semantic-type-expander');
                    }
                });
            }
            // Run on load and after delays to catch dynamically added elements
            styleSemanticTypeExpander();
            setTimeout(styleSemanticTypeExpander, 100);
            setTimeout(styleSemanticTypeExpander, 300);
            setTimeout(styleSemanticTypeExpander, 500);
            setTimeout(styleSemanticTypeExpander, 1000);
            // Watch for DOM changes
            const observer = new MutationObserver(styleSemanticTypeExpander);
            observer.observe(document.body, { childList: true, subtree: true });
        </script>
    """, unsafe_allow_html=True)
    
    qdf: pd.DataFrame = st.session_state.query_df.copy()
    
    # Create two-column layout: Edit Semantic Types in col1, rest in col2
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Section for editing semantic types (column headers)
        with st.expander("📝 Edit Semantic Types", expanded=True):
            st.caption("💡 **Hint:** Column headers represent semantic types for matching with data lake tables. Edit them to specify the semantic meaning of each column.")
            
            # Add Column button
            add_col = st.button("➕ Add Column", use_container_width=True)
            if add_col:
                new_semantic_type = f"SemanticType_{len(qdf.columns) + 1}"
                qdf[new_semantic_type] = ""
                st.session_state.query_df = qdf
                st.rerun()
            
            semantic_types = {}
            cols_per_row = 1  # Single column layout for semantic type inputs
            col_names_list = list(qdf.columns)
            
            # Use a version key based on semantic types to reset widgets when columns change
            semantic_types_hash = hash(tuple(col_names_list))
            version_key = f"semantic_type_version_{semantic_types_hash}"
            
            # Initialize session state for storing edits
            semantic_type_state_key = "semantic_type_edits"
            if semantic_type_state_key not in st.session_state:
                st.session_state[semantic_type_state_key] = {}
            
            # If semantic types changed, reset the edits to match current semantic types
            if version_key not in st.session_state or st.session_state[version_key] != semantic_types_hash:
                st.session_state[semantic_type_state_key] = {i: name for i, name in enumerate(col_names_list)}
                st.session_state[version_key] = semantic_types_hash
            
            for i in range(0, len(col_names_list), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, semantic_type in enumerate(col_names_list[i:i+cols_per_row]):
                    with cols[j]:
                        # Use column index for stable key generation
                        col_idx = i + j
                        # Get value from session state, defaulting to current semantic type
                        current_value = st.session_state[semantic_type_state_key].get(col_idx, semantic_type)
                        # Include version in key to reset widget when semantic types change
                        new_semantic_type = st.text_input(
                            f"Semantic Type {col_idx + 1}",
                            value=current_value,
                            key=f"semantic_type_idx_{col_idx}_v{semantic_types_hash}",
                            label_visibility="visible",
                            help=f"Edit semantic type (current: {semantic_type})"
                        )
                        # Store in session state
                        st.session_state[semantic_type_state_key][col_idx] = new_semantic_type
                        semantic_types[semantic_type] = new_semantic_type
            
            apply_semantic_types = st.button("✅ Apply Semantic Type Changes", type="primary", use_container_width=True)
            if apply_semantic_types:
                # Rename columns if semantic types changed
                rename_map = {old: new for old, new in semantic_types.items() if old != new and new.strip()}
                if rename_map:
                    qdf = qdf.rename(columns=rename_map)
                    st.session_state.query_df = qdf
                    # Clear version key to force reset on next render
                    if version_key in st.session_state:
                        del st.session_state[version_key]
                    st.session_state.query_ready = False
                    st.session_state.query_submitted = False
                    st.rerun()
    
    with col2:
        st.success(f"Query table ready: Rows {qdf.shape[0]} • Cols {qdf.shape[1]}")
        
        # Use data_editor to make the table editable
        edited_df = st.data_editor(
            qdf,
            use_container_width=True,
            height=420,
            num_rows="dynamic",
            column_config={
                col: st.column_config.TextColumn(
                    col,
                    help=f"Semantic type: {col}"
                )
                for col in qdf.columns
            }
        )
        
        # Update session state if dataframe was edited
        if not edited_df.equals(qdf):
            st.session_state.query_df = edited_df
            # Update semantic type edits if columns changed
            if list(edited_df.columns) != list(qdf.columns):
                st.session_state.semantic_type_edits = {i: name for i, name in enumerate(edited_df.columns)}
            st.session_state.query_ready = False
            st.session_state.query_submitted = False
        
        if not st.session_state.query_submitted:
            st.info("Click **Submit query table**, then proceed to **Retrieved Join Paths**.")
        
        # Top k join paths slider
        st.session_state.ret_top_k = st.slider(
            "Top‑k join paths",
            min_value=1,
            max_value=50,
            value=int(st.session_state.ret_top_k),
            step=1,
            key="ret_top_k_slider",
        )
        
        # Number of paths to display slider (for comparison view)
        if "num_display_paths" not in st.session_state:
            st.session_state.num_display_paths = 20
        st.session_state.num_display_paths = st.slider(
            "Number of paths to display",
            min_value=10,
            max_value=30,
            value=int(st.session_state.num_display_paths),
            step=2,
            key="num_display_paths_slider",
            help="Control how many join paths to show in the comparison view (10-30)",
        )
        
        # Navigation and action buttons
        c_nav1, c_nav2, c_nav3, c_nav4 = st.columns([0.25, 0.25, 0.25, 0.25])
        with c_nav1:
            if st.button("⬅️ Back", use_container_width=True):
                goto_stage(1)
        with c_nav2:
            add_row = st.button("➕ Add Row", use_container_width=True)
        with c_nav3:
            reset_query = st.button("🔄 Reset query table", use_container_width=True)
        with c_nav4:
            submit_query = st.button("📤 Submit query table", type="primary", use_container_width=True)
        
        # Handle add row button
        if add_row:
            new_row = pd.DataFrame({col: [""] for col in qdf.columns}, index=[len(qdf)])
            qdf = pd.concat([qdf, new_row], ignore_index=True)
            st.session_state.query_df = qdf
            st.rerun()
        
        # Handle reset query button
        if reset_query:
            st.session_state.query_df = bundled_query()
            st.session_state.query_ready = False
            st.session_state.query_submitted = False
            st.rerun()
        
        # Handle submit query button
        if submit_query:
            st.session_state.query_ready = True
            st.session_state.query_submitted = True
            
            # Show progress bar for 1 second
            progress_bar = st.progress(0, text="Submitting query table...")
            num_steps = 20  # 20 steps for smooth animation
            duration = 1.0  # 1 second total
            step_duration = duration / num_steps
            
            for i in range(1, num_steps + 1):
                progress = i / num_steps
                progress_bar.progress(progress, text="Submitting query table...")
                time.sleep(step_duration)
            
            progress_bar.empty()
            goto_stage(3)

