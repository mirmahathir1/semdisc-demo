from __future__ import annotations

from typing import Dict, Optional

import pandas as pd
import streamlit as st

from utils import (
    bundled_datalake,
    ensure_nonempty_tables,
    goto_stage,
    simulate_preprocessing,
    token_for_uploaded_files,
)


def render() -> None:
    """Render Stage 0: Upload & Preprocess."""
    left_col, right_col = st.columns([0.62, 0.38], gap="large")

    with right_col:
        st.subheader("Preprocessing updates")
        updates_placeholder = st.empty()

        def _render_preprocess_updates() -> None:
            log = list(st.session_state.get("preprocess_log", []) or [])
            with updates_placeholder.container():
                if log:
                    st.code("\n".join(log[-200:]), language="text")
                else:
                    st.caption("No preprocessing run yet.")

        _render_preprocess_updates()

    with left_col:
        st.subheader("Upload and Preprocess")
        uploaded_files = st.file_uploader(
            "Data lake folder (upload CSV files)",
            type=["csv"],
            accept_multiple_files=True,
            help="Streamlit can't directly upload a folder. Upload any CSV file(s) from the folder to unlock preprocessing (the demo run uses the bundled sample data).",
        )

        st.subheader("Hyperparameters")
        c3, c4, c5 = st.columns(3)
        with c3:
            st.session_state.hp_simhash_size = st.slider(
                "SimHash size (bits)",
                min_value=32,
                max_value=512,
                value=int(st.session_state.hp_simhash_size),
                step=32,
            )
            st.session_state.hp_join_edge_threshold = st.slider(
                "Join edge threshold",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.hp_join_edge_threshold),
                step=0.01,
            )
        with c4:
            st.session_state.hp_num_hashes_minhash = st.slider(
                "Number of hashes for MinHash",
                min_value=16,
                max_value=512,
                value=int(st.session_state.hp_num_hashes_minhash),
                step=16,
            )
            st.session_state.hp_semantic_type_similarity_threshold = st.slider(
                "Semantic type similarity threshold",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.hp_semantic_type_similarity_threshold),
                step=0.01,
            )
        with c5:
            st.session_state.hp_simple_path_max_length = st.slider(
                "Simple path max length",
                min_value=1,
                max_value=5,
                value=int(st.session_state.hp_simple_path_max_length),
                step=1,
            )

        preprocess_disabled = not uploaded_files
        if preprocess_disabled:
            st.info("Upload your data lake CSV files to enable preprocessing.")

        preprocess_clicked = st.button(
            "Preprocess",
            type="primary",
            use_container_width=True,
            disabled=preprocess_disabled,
        )
        reset_clicked = st.button("Reset processed data", use_container_width=True)

    if reset_clicked:
        st.session_state.processed_ready = False
        st.session_state.tables = {}
        st.session_state.datalake_token = None

    uploaded_token: Optional[str] = token_for_uploaded_files(uploaded_files)

    if uploaded_token is not None and st.session_state.datalake_token != uploaded_token:
        # New upload selected; invalidate prior processed results.
        st.session_state.processed_ready = False
        st.session_state.tables = {}
        st.session_state.datalake_token = uploaded_token

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
        simulate_preprocessing(steps=steps, container=updates_placeholder)
        _render_preprocess_updates()

        if not uploaded_files:
            st.warning("Please upload your data lake CSV files before preprocessing.")
            return
        tables = bundled_datalake()
        tables = ensure_nonempty_tables(tables)
        if not tables:
            st.error("No tables found in the data lake.")
            return
        st.session_state.tables = tables
        st.session_state.processed_ready = True
        # Seed downstream defaults from preprocessing hyperparameters.
        st.session_state.nav_min_edge_score = float(st.session_state.hp_join_edge_threshold)
        st.session_state.nav_max_hops = int(st.session_state.hp_simple_path_max_length)
        st.session_state.ret_min_edge_score = float(st.session_state.hp_join_edge_threshold)
        st.session_state.ret_max_hops = int(st.session_state.hp_simple_path_max_length)

    if st.session_state.processed_ready:
        st.success(f"Processed data ready: {len(st.session_state.tables)} tables.")
    else:
        st.info("Click **Preprocess** to load and prepare the data lake.")

    if st.session_state.processed_ready:
        if st.button("Proceed to next stage", type="primary", use_container_width=True):
            goto_stage(1)

