# Implementation Plan: UI Improvements for Semantic Discovery Demo

## Overview
This plan outlines the implementation of UI enhancements for visualizing join paths and semantic annotations in both the offline indexing phase and the query results phase.

> **Important**: This is a demo application. All data shown is simulated and hardcoded—no real backend computations or LLM calls are made. The goal is to create a convincing visualization, not to process real data.

---

## Data Structures

### Semantic Types
- Semantic types are **strings** derived from an LLM annotation (hardcoded for demo)
- Each column in a data lake table has one semantic type
- Example: Column `plan_id` → Semantic Type `"Plan Unique Identifier"`

### Join Paths
- A join path is a **list of tables with edges** between them
- Each edge contains the **column pair** involved in the join
- Data structure example:
  ```
  JoinPath = {
    tables: [Table1, Table2, Table3, ...],
    edges: [
      { from: Table1.column_a, to: Table2.column_b },
      { from: Table2.column_c, to: Table3.column_d },
      ...
    ]
  }
  ```

### Join Path Count
- The demo data should yield **at least 30 discoverable join paths**
- The Navigate page computes join paths dynamically (top‑k per start table, then dedup + global sort)
- **No “10–30 paths” slider** is currently used on the Navigate page (all computed paths are shown in list/heatmap views)

---

## Phase 1: Expand Demo Data & Semantic Types

**Goal**: Expand the bundled demo data to support 30 join paths and add hardcoded semantic type annotations for all columns.

### Tasks
- [x] Expand `bundled_datalake()` in `utils.py` to include ~12-15 tables with more interconnections
- [x] Create a `SEMANTIC_TYPES` dictionary mapping `(table_name, column_name)` → semantic type string
- [x] Ensure the expanded data produces at least 30 unique join paths

**Verification (current `utils.py`)**:
- 15 bundled tables
- 30 undirected join edges (at `min_edge_score=0.10`)
- Default max hops is **5** (via `hp_simple_path_max_length` in `app.py`)
- ~14,864 unique simple paths (up to **5** hops; exhaustive enumeration over the inferred join graph)
- ~150 unique join paths in the Navigate UI list (deduped after taking top‑k per start table, with `k=10`)

### Deliverables
- Updated `bundled_datalake()` function with expanded e-commerce tables
- New `SEMANTIC_TYPES` constant in `utils.py`

### How to Test
1. Run the app: `streamlit run app.py`
2. Click **"Preprocess"** on the Upload page (no ZIP needed — uses bundled data)
3. Go to **"Navigate Processed Data"** page
4. **Verify**: You should see more tables listed (12-15 instead of 6)
5. Select different start tables and verify join paths are generated
6. **Verify**: At least 30 join paths should be discoverable across all start tables

---

## Phase 2: Display Semantic Types in Offline View

**Goal**: Add a UI section on the "Navigate Processed Data" page to display discovered semantic types for each table's columns.

### Tasks
- [x] Add a "Semantic types" view on the Navigate page (implemented as a tab)
- [x] Display a table/grid showing: Table Name | Column Name | Semantic Type
- [x] Style semantic types as readable badges (custom HTML table for consistent styling)

### Deliverables
- New "Semantic types" tab in `page_navigate.py`

### How to Test
1. Run the app and preprocess the bundled data
2. Go to **"Navigate Processed Data"** page
3. **Verify**: You see a **"Semantic types"** tab
4. **Verify**: Each column from each table has a semantic type displayed
5. **Verify**: The semantic types are readable strings (e.g., "Customer Identifier", "Order Date")

---

## Phase 3: Heatmap Visualization Component

**Goal**: Create a reusable heatmap component for visualizing join paths, with click interaction to show the selected path in a side panel.

### Specifications
- **X-axis**: Number of nodes in the join path
- **Y-axis**: Tuple count bins (10 bins, dynamically computed)
- **Color**: Single red gradient (white → dark red), based on **how many join paths fall in that cell** (path-count)
- **Hover**: Show **join path count**, tuple-count range for the bin, and node count
- **Click**: Select a **heatmap cell** (may represent multiple join paths) and display them in a side panel

### Tasks
- [x] Create `components/heatmap.py` and `components/heatmap_component/index.html` to render a clickable heatmap (custom HTML/JS Streamlit component; no build tooling)
- [x] Implement 10-bin calculation based on min/max tuple counts
- [x] Implement a red gradient color scale (white → dark red)
- [x] Add hover tooltip showing join path count + tuple-count range + node count
- [x] Add click handler that updates Streamlit session state with the selected **cell** (`best_path_key` + `path_keys`)
- [x] Add a side panel showing the selected cell metadata and Graphviz diagrams for the paths in that cell (expanders)

### Deliverables
- New `components/heatmap.py` module
- Heatmap integrated into Navigate page with toggle

### How to Test
1. Run the app and preprocess the bundled data
2. Go to **"Navigate Processed Data"** page
3. Click the **"Join paths"** tab
4. **Verify**: You see a toggle switch "List view / Heatmap view"
4. Click the toggle to switch to **Heatmap View**
5. **Verify**: Heatmap appears with X-axis (node count) and Y-axis (tuple bins)
6. **Verify**: Cells are colored in red gradient (darker = more join paths in that cell)
7. Hover over a cell → **Verify**: Tooltip shows "Join paths: N | Tuples: A–B | Nodes: K"
8. Click a cell → **Verify**: The cell highlights and the right-side panel shows the selected cell + one or more path diagrams (expandable)

---

## Phase 4: Query Table Interactive Selection

**Goal**: Show the query table on the **Retrieved Join Paths** page and add row/column selection on that query table with visual highlighting and mutual exclusivity.

### Specifications
- **Query table location**: The query table is rendered on the **Retrieved Join Paths** page (this is the table that receives selection + highlighting).
- **Row selection**: Click a row → row turns blue (in the retrieved page query table), entire matching join paths highlight blue
- **Column selection**: Click a column header → column highlights (in the retrieved page query table), matching nodes turn green
- Selecting a row keeps column selection, and vice versa
- **Default state**: First row is pre-selected on page load

### Tasks
- [ ] Ensure the query table is displayed on the **Retrieved Join Paths** page
- [ ] Modify the retrieved-page query table display to support row click selection
- [ ] Modify the retrieved-page query table display to support column header click selection
- [ ] Add CSS styling for selected row (blue background)
- [ ] Add CSS styling for selected column (green header)
- [ ] Implement session state for `selected_row` and `selected_column`
- [ ] Pre-select first row on initial page load

### Deliverables
- Updated `page_retrieved_paths.py` to render the query table and apply selection + highlighting there
- Updated `page_query_input.py` to store the submitted query table in session state for use on the retrieved page

### How to Test
1. Run the app, preprocess data, and submit a query table
2. Go to **"Retrieved Join Paths"** page
3. **Verify**: The query table is visible on the Retrieved page
4. **Verify**: First row is pre-selected (highlighted blue) by default (on the Retrieved page query table)
5. Click a different row → **Verify**: That row turns blue, previous selection clears (on the Retrieved page query table)
6. Click a column header → **Verify**: Column highlights green, row selection stays (on the Retrieved page query table)
7. Click a row again → **Verify**: Row turns blue, column selection stays (on the Retrieved page query table)
---

## Phase 5: Join Path Comparison View with Pagination

**Goal**: Display 2 join paths side-by-side for comparison, with pagination to navigate through all paths.

### Specifications
- **Layout**: Query table on left, 2 join paths side-by-side on right
- **Pagination**: Navigate through paths 2 at a time (Page 1: paths 1-2, Page 2: paths 3-4, etc.)
- **Highlighting**: Green nodes for column overlap (exact match), blue entire path for row match

### Tasks
- [ ] Redesign Retrieved Paths page layout: left column (query table), right column (2 join paths)
- [ ] Implement pagination controls (Previous / Next buttons, page indicator)
- [ ] Display 2 join paths per page using Graphviz
- [ ] Implement green node highlighting for column value matches
- [ ] Implement blue path highlighting for full row matches
- [ ] Add slider to control total number of displayed paths (10-30)

### Deliverables
- Redesigned `page_retrieved_paths.py` with comparison view
- Pagination component
- Highlighting logic integrated with Phase 4 selection state

### How to Test
1. Run the app, preprocess data, and submit a query table
2. Go to **"Retrieved Join Paths"** page
3. **Verify**: Layout shows query table on left, 2 join paths on right
4. **Verify**: Pagination shows "Page 1 of N" with Previous/Next buttons
5. Click **Next** → **Verify**: Shows next 2 join paths
6. Click **Previous** → **Verify**: Returns to previous 2 paths
7. With a row selected (blue), **Verify**: Join paths containing that full row are highlighted blue
8. Select a column instead → **Verify**: Nodes containing matching column values turn green
9. Adjust the slider (10-30) → **Verify**: Total number of paths changes, pagination updates

---

## Phase 6: Heatmap Integration & Final Polish

**Goal**: Integrate heatmap into both offline (Navigate) and query results (Retrieved Paths) pages with consistent behavior.

### Tasks
- [ ] Add heatmap toggle to Retrieved Paths page (reuse component from Phase 3)
- [ ] Ensure both heatmaps use identical configuration (bins, colors, tooltips)
- [ ] Connect heatmap click to update the comparison view (show clicked path)
- [ ] Final styling pass for consistent look and feel
- [ ] Test all interactions end-to-end

### Deliverables
- Heatmap on both Navigate and Retrieved Paths pages
- Consistent toggle behavior
- Polished UI

### How to Test
1. **Test Navigate Page Heatmap**:
   - Go to Navigate page, switch to Heatmap view
   - Click cells, verify join path shows in side panel
   - Switch back to List view, verify it works

2. **Test Retrieved Paths Page Heatmap**:
   - Go to Retrieved Paths page, switch to Heatmap view
   - Click a cell → **Verify**: The comparison view updates to show that path
   - Hover cells → **Verify**: Tooltips show correct data
   - Switch back to List view, verify pagination still works

3. **Test End-to-End Flow**:
   - Start fresh: Preprocess → Navigate (check semantic types, heatmap) → Query Input → Retrieved Paths
   - Select rows/columns, verify highlighting
   - Use pagination, verify path comparison
   - Use heatmap, verify click interaction
   - **Verify**: All features work together without conflicts

---

## Technical Considerations

### Frontend
- **Framework**: Streamlit (existing)
- **Charting**: Graphviz for join paths; custom HTML/JS Streamlit component for the heatmap (no Plotly build step)
- **State management**: Streamlit session state for selections, pagination, view toggles
- **Styling**: Custom CSS via `st.markdown(unsafe_allow_html=True)`
- **Navigation**: Sidebar is hidden; navigation uses an in-page stepper/progress indicator + buttons

### Files to Modify/Create
| Phase | Files |
|-------|-------|
| 1 | `utils.py` |
| 2 | `page_navigate.py` |
| 3 | `components/heatmap.py` (new), `page_navigate.py` |
| 4 | `page_query_input.py`, `page_retrieved_paths.py` |
| 5 | `page_retrieved_paths.py` |
| 6 | `page_navigate.py`, `page_retrieved_paths.py` |

---

## Resolved Questions Reference

### Tech Stack
- **Framework**: Streamlit (Python) — existing
- **Charting**: Graphviz for paths; custom HTML/JS component for heatmap

### Data & Visualization
- **Tables**: `Dict[str, pd.DataFrame]`
- **Join paths**: Graphviz linear chain visualization
- **Domain**: E-commerce (customers, orders, products, etc.)
- **Scale**: Expand to ~12-15 tables for 30 join paths

### Heatmap
- **Bins**: 10 bins, dynamically computed
- **Color**: Single red gradient (white → dark red)
- **Click**: Selects a heatmap cell and shows one or more paths in a side panel

### Interaction
- **Row selection**: Blue, highlights entire path
- **Column selection**: Green, highlights matching nodes
- **Default**: First row pre-selected
- **Node clicks**: Not supported
