from __future__ import annotations

import pandas as pd
import streamlit as st

from utils import bundled_query, goto_stage


def render() -> None:
    """Render Stage 2: Input Query Table."""
    st.subheader("Input Query Table")
    
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
            
            apply_semantic_types = st.button("Apply Semantic Type Changes", type="primary", use_container_width=True)
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
        
        # Navigation and action buttons
        c_nav1, c_nav2, c_nav3, c_nav4 = st.columns([0.25, 0.25, 0.25, 0.25])
        with c_nav1:
            if st.button("Back to previous stage", use_container_width=True):
                goto_stage(1)
        with c_nav2:
            add_row = st.button("➕ Add Row", use_container_width=True)
        with c_nav3:
            reset_query = st.button("Reset query table", use_container_width=True)
        with c_nav4:
            submit_query = st.button("Submit query table", type="primary", use_container_width=True)
        
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
            goto_stage(3)

