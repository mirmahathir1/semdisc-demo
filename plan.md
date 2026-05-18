# Flask + Vue Refactor Plan

## Goal

Refactor the current Python-only Streamlit demo into a Flask + Vue.js application:

- `backend/` owns all data loading, preprocessing simulation, join discovery, path materialization, serialization, and API endpoints.
- `frontend/` owns all presentation, navigation, interactive tables, heatmaps, graph rendering, and user workflow state.
- All Streamlit code, Streamlit custom component protocol code, and Streamlit dependencies are removed.

## Current State

The existing app is a Streamlit application with root-level page modules:

- `app.py` initializes Streamlit session state, renders global header controls, handles preprocessing, and routes between stages.
- `page_navigate.py` renders processed data navigation, semantic type badges, heatmap selection, path details, and graph views.
- `page_query_input.py` renders editable query table and semantic type inputs.
- `page_retrieved_paths.py` renders retrieved paths from the query, heatmaps, path detail views, and materialized tables.
- `page_upload_preprocess.py` is an older Streamlit upload/preprocess page.
- `components/heatmap.py` and `components/query_table.py` are Streamlit Python wrappers around HTML components.
- `components/*_component/index.html` use the Streamlit component postMessage protocol.
- `utils.py` mixes reusable data/join logic with Streamlit-only helpers such as cache decorators, progress UI, headers, Graphviz rendering, and navigation.
- `requirements.txt` includes `streamlit>=1.31` and `pandas>=2.0`.

## Target Repository Layout

```text
backend/
  app.py
  config.py
  requirements.txt
  semdisc/
    __init__.py
    models.py
    data_loader.py
    join_graph.py
    paths.py
    preprocessing.py
    serialization.py
    services.py
    api/
      __init__.py
      health.py
      datasets.py
      query.py
      graph.py
      paths.py
  tests/
    test_data_loader.py
    test_join_graph.py
    test_paths.py
    test_api.py

frontend/
  package.json
  vite.config.js
  index.html
  src/
    main.js
    App.vue
    api/client.js
    stores/workflow.js
    router/index.js
    views/
      NavigateView.vue
      QueryInputView.vue
      RetrievedPathsView.vue
    components/
      AppHeader.vue
      StageStepper.vue
      CsvUpload.vue
      HyperparameterControls.vue
      DataTable.vue
      EditableQueryTable.vue
      SemanticTypeEditor.vue
      JoinHeatmap.vue
      JoinGraph.vue
      PathList.vue
      PathDetail.vue
      MaterializedTable.vue
    styles/
      base.css
      layout.css

datalake/
README.md
plan.md
```

## Backend Plan

1. Create the Flask app skeleton.
   - Add `backend/app.py` with an app factory.
   - Add `backend/semdisc/api/` blueprints for health, dataset, query, graph, and path endpoints.
   - Add CORS support for local Vue development.
   - Add JSON error handling with consistent `{ "error": { "code": "...", "message": "..." } }` responses.

2. Extract Streamlit-independent domain logic from `utils.py`.
   - Move `EdgeInfo` and related response types into `backend/semdisc/models.py`.
   - Move `SEMANTIC_TYPES`, `normalize_cell`, `jaccard`, `best_join_key_and_score`, `best_semantic_join_keys_and_score`, `build_join_graph`, and `attach_query_node` into `backend/semdisc/join_graph.py`.
   - Move `simple_paths_topk`, `diversify_paths_by_node_count`, `format_path`, `path_to_dot`, `path_to_dot_highlighted`, and query-path helpers into `backend/semdisc/paths.py`.
   - Move `materialize_path` into `backend/semdisc/services.py` or `backend/semdisc/paths.py`, depending on how much orchestration it needs.
   - Move `bundled_datalake`, `bundled_query`, `read_csv_bytes`, `load_datalake_from_zip`, `ensure_nonempty_tables`, and `drop_fully_blank_rows` into `backend/semdisc/data_loader.py`.
   - Replace `@st.cache_data` with Flask-side caching where useful, such as `functools.lru_cache` for bundled demo data or a request/session cache for uploaded data.

3. Remove all backend Streamlit coupling.
   - Delete imports of `streamlit` and `streamlit.components.v1`.
   - Delete `render_graphviz_fixed_height`, `simulate_preprocessing` UI code, `render_brand_header`, `render_stage_stepper`, and `goto_stage`.
   - Replace Streamlit session state with explicit backend state.
   - Keep backend state minimal: either in-memory per-process demo state or a simple session/dataset id returned by upload/preprocess endpoints.

4. Define API payloads around JSON-friendly structures.
   - Serialize pandas DataFrames as `{ "columns": [...], "rows": [{...}] }`.
   - Serialize edges as `{ "left": "...", "right": "...", "key": "...", "rightKey": "...", "score": 0.95, "joinType": "equi" }`.
   - Serialize graphs as `{ "nodes": [...], "edges": [...] }`, not Graphviz DOT as the primary contract.
   - Return DOT only as an optional field if the frontend graph component initially needs it.

5. Implement backend endpoints.
   - `GET /api/health`: service readiness.
   - `GET /api/demo/datalake`: load bundled demo tables.
   - `GET /api/demo/query`: load bundled query table.
   - `POST /api/datasets`: accept CSV files or a ZIP, sanitize tables, return a dataset id and table summaries.
   - `POST /api/preprocess`: accept dataset id and hyperparameters, compute graph/path indexes, return preprocessing status and summaries.
   - `GET /api/datasets/<dataset_id>/tables`: table list and schema metadata.
   - `GET /api/datasets/<dataset_id>/tables/<table_name>`: paginated table preview.
   - `GET /api/datasets/<dataset_id>/semantic-types`: semantic type metadata by table.
   - `GET /api/datasets/<dataset_id>/graph`: join graph for selected thresholds.
   - `GET /api/datasets/<dataset_id>/paths`: join paths for navigation heatmap.
   - `POST /api/query`: submit the editable query table and return a query id.
   - `GET /api/query/<query_id>/paths`: retrieve top-k paths attached to the query node.
   - `POST /api/materialize`: materialize a selected path and return a paginated table.

6. Add backend tests.
   - Unit-test join scoring, semantic join selection, graph construction, path ranking, and materialization.
   - API-test health, bundled data loading, preprocessing, graph retrieval, query submission, path retrieval, and materialization.
   - Add regression tests for uploaded CSV/ZIP parsing and blank-row cleanup.

## Frontend Plan

1. Create a Vite Vue app in `frontend/`.
   - Use Vue 3 with the Composition API.
   - Use Vue Router for the three-stage workflow.
   - Use Pinia or a small composable store for dataset id, query id, hyperparameters, selected row/column, selected heatmap cell, and selected path.

2. Rebuild the Streamlit stages as Vue views.
   - `NavigateView.vue`: replaces `page_navigate.py`.
   - `QueryInputView.vue`: replaces `page_query_input.py`.
   - `RetrievedPathsView.vue`: replaces `page_retrieved_paths.py`.
   - The old upload/preprocess page remains removed; upload and preprocessing controls live in the header, matching the current Streamlit app behavior.

3. Rebuild shared UI as Vue components.
   - `AppHeader.vue`: SemDisc brand, upload control, hyperparameter controls, preprocess/reset actions.
   - `StageStepper.vue`: replaces `render_stage_stepper`.
   - `DataTable.vue`: reusable paginated table preview.
   - `EditableQueryTable.vue`: editable query rows and cells.
   - `SemanticTypeEditor.vue`: editable semantic type/column names.
   - `JoinHeatmap.vue`: replaces the Streamlit heatmap component with native Vue events and props.
   - `JoinGraph.vue`: renders the graph from JSON nodes/edges using a browser graph library such as Cytoscape.js, D3, or a DOT renderer if preserving the current Graphviz layout is important.
   - `PathList.vue`, `PathDetail.vue`, and `MaterializedTable.vue`: path browsing and selected path inspection.

4. Move presentation behavior out of backend logic.
   - Heatmap binning can live in the frontend if it is purely visual.
   - Path selection, row/column selection, table editing, button styling, and route navigation live only in Vue.
   - Backend returns data; frontend decides how it is displayed.

5. Add frontend tests.
   - Component tests for heatmap selection, query table editing, semantic type renaming, and path selection.
   - API client tests with mocked Flask responses.
   - End-to-end smoke tests for preprocess -> navigate -> edit query -> retrieve paths -> materialize selected path.

## Streamlit Removal Plan

Remove these files after equivalent Flask/Vue functionality exists:

- `app.py`
- `page_upload_preprocess.py`
- `page_navigate.py`
- `page_query_input.py`
- `page_retrieved_paths.py`
- `components/heatmap.py`
- `components/query_table.py`
- `components/__init__.py`
- `components/heatmap_component/index.html`
- `components/query_table_component/index.html`

Remove or rewrite these Streamlit-specific code paths:

- Every `import streamlit as st`.
- Every `import streamlit.components.v1 as components`.
- Every `st.session_state` access.
- Every `@st.cache_data` decorator.
- Every `st.*` UI call, including `st.markdown`, `st.dataframe`, `st.data_editor`, `st.button`, `st.slider`, `st.file_uploader`, `st.progress`, `st.spinner`, `st.columns`, `st.container`, `st.download_button`, `st.rerun`, and `st.experimental_rerun`.
- Streamlit component postMessage protocol calls in the HTML component files.
- Streamlit command references in docs and development scripts.

Update dependencies:

- Replace root `requirements.txt` with backend-only dependencies or move it to `backend/requirements.txt`.
- Remove `streamlit>=1.31`.
- Keep `pandas>=2.0`.
- Add Flask dependencies, likely `flask`, `flask-cors`, and test dependencies such as `pytest`.
- Add frontend dependencies in `frontend/package.json`.

## Migration Sequence

1. Add the new folder structure without deleting existing Streamlit files.
2. Extract pure data and join logic into `backend/semdisc/`.
3. Add tests around extracted backend logic and verify parity with current demo behavior.
4. Implement Flask app factory, blueprints, serializers, and API tests.
5. Scaffold Vue app and API client.
6. Build the shared Vue layout, header, stage stepper, and workflow store.
7. Implement `NavigateView.vue` against Flask graph/path/table endpoints.
8. Implement `QueryInputView.vue` against bundled-query and query-submission endpoints.
9. Implement `RetrievedPathsView.vue` against query path and materialization endpoints.
10. Run an end-to-end workflow locally with Flask and Vite.
11. Delete all Streamlit files and component protocol code.
12. Update `README.md` with separate backend and frontend setup/run instructions.
13. Run backend tests, frontend tests, and an end-to-end smoke test before final cleanup.

## Acceptance Criteria

- The app runs with Flask serving APIs from `backend/` and Vue serving UI from `frontend/`.
- No production code imports Streamlit.
- `requirements.txt` or `backend/requirements.txt` does not include Streamlit.
- Root-level Streamlit page files and `components/` Streamlit wrappers are deleted.
- The demo workflow still supports preprocessing, data navigation, query table editing, join path retrieval, heatmap filtering, graph/path inspection, and materialized table viewing.
- Backend logic is covered by focused unit and API tests.
- The README documents the new stack, setup commands, dev server commands, and cleanup from the old Streamlit app.

## Risks And Decisions To Make

- Backend state: choose in-memory demo state for speed, or persisted dataset/query ids for multi-user correctness.
- Graph rendering: choose a native browser graph library or preserve DOT generation temporarily.
- Upload behavior: decide whether to accept multiple CSV files, ZIP files, or both.
- Large datasets: add pagination and avoid returning full materialized tables in a single response.
- Cache invalidation: define when uploaded datasets, query submissions, and preprocessed graphs expire.
