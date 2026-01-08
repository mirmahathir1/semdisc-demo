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
- **Hardcode 30 join paths** in the code
- Provide a **slider** to control how many are displayed to the user (range: 10–30)

---

## Phase 1: Expand Demo Data & Semantic Types

**Goal**: Expand the bundled demo data to support 30 join paths and add hardcoded semantic type annotations for all columns.

### Tasks
- [ ] Expand `bundled_datalake()` in `utils.py` to include ~12-15 tables with more interconnections
- [ ] Create a `SEMANTIC_TYPES` dictionary mapping `(table_name, column_name)` → semantic type string
- [ ] Ensure the expanded data produces at least 30 unique join paths

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
- [ ] Add a new expandable section "Semantic Types" on the Navigate page
- [ ] Display a table/grid showing: Table Name | Column Name | Semantic Type
- [ ] Style the semantic types with subtle background colors or badges

### Deliverables
- New "Semantic Types" section in `page_navigate.py`

### How to Test
1. Run the app and preprocess the bundled data
2. Go to **"Navigate Processed Data"** page
3. **Verify**: You see a new "Semantic Types" section (expandable or always visible)
4. **Verify**: Each column from each table has a semantic type displayed
5. **Verify**: The semantic types are readable strings (e.g., "Customer Identifier", "Order Date")

---

## Phase 3: Heatmap Visualization Component

**Goal**: Create a reusable heatmap component for visualizing join paths, with click interaction to show the selected path in a side panel.

### Specifications
- **X-axis**: Number of nodes in the join path
- **Y-axis**: Tuple count bins (10 bins, dynamically computed)
- **Color**: Single red gradient (white → dark red)
- **Hover**: Show column overlap count and tuple count
- **Click**: Display selected join path in side panel

### Tasks
- [ ] Create `components/heatmap.py` with a function to render heatmap using Plotly or custom HTML/JS
- [ ] Implement 10-bin calculation based on min/max tuple counts
- [ ] Implement red gradient color scale
- [ ] Add hover tooltip with column overlap count and tuple count
- [ ] Add click handler that updates Streamlit session state with selected path
- [ ] Create side panel layout to show selected join path (Graphviz)

### Deliverables
- New `components/heatmap.py` module
- Heatmap integrated into Navigate page with toggle

### How to Test
1. Run the app and preprocess the bundled data
2. Go to **"Navigate Processed Data"** page
3. **Verify**: You see a toggle switch "List View / Heatmap View"
4. Click the toggle to switch to **Heatmap View**
5. **Verify**: Heatmap appears with X-axis (node count) and Y-axis (tuple bins)
6. **Verify**: Cells are colored in red gradient (darker = more tuples)
7. Hover over a cell → **Verify**: Tooltip shows "Column overlap: X, Tuple count: Y"
8. Click a cell → **Verify**: The corresponding join path appears in the side panel as a Graphviz diagram

---

## Phase 4: Query Table Interactive Selection

**Goal**: Add row and column selection to the query table with visual highlighting and mutual exclusivity.

### Specifications
- **Row selection**: Click a row → row turns blue, entire matching join paths highlight blue
- **Column selection**: Click a column header → column highlights, matching nodes turn green
- **Mutual exclusivity**: Selecting a row clears column selection, and vice versa
- **Default state**: First row is pre-selected on page load

### Tasks
- [ ] Modify query table display to support row click selection
- [ ] Modify query table display to support column header click selection
- [ ] Add CSS styling for selected row (blue background)
- [ ] Add CSS styling for selected column (green header)
- [ ] Implement session state for `selected_row` and `selected_column`
- [ ] Implement mutual exclusivity logic
- [ ] Pre-select first row on initial page load

### Deliverables
- Updated `page_query_input.py` with interactive selection
- Updated `page_retrieved_paths.py` to read selection state

### How to Test
1. Run the app, preprocess data, and submit a query table
2. Go to **"Retrieved Join Paths"** page
3. **Verify**: First row is pre-selected (highlighted blue) by default
4. Click a different row → **Verify**: That row turns blue, previous selection clears
5. Click a column header → **Verify**: Column highlights green, row selection clears
6. Click a row again → **Verify**: Row turns blue, column selection clears
7. **Verify**: Only one selection (row OR column) is active at any time

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
- **Charting**: Graphviz for join paths, Plotly/custom HTML for heatmap
- **State management**: Streamlit session state for selections, pagination, view toggles
- **Styling**: Custom CSS via `st.markdown(unsafe_allow_html=True)`

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
- **Charting**: Graphviz for paths, Plotly/HTML for heatmap

### Data & Visualization
- **Tables**: `Dict[str, pd.DataFrame]`
- **Join paths**: Graphviz linear chain visualization
- **Domain**: E-commerce (customers, orders, products, etc.)
- **Scale**: Expand to ~12-15 tables for 30 join paths

### Heatmap
- **Bins**: 10 bins, dynamically computed
- **Color**: Single red gradient (white → dark red)
- **Click**: Shows path in side panel

### Interaction
- **Row selection**: Blue, highlights entire path
- **Column selection**: Green, highlights matching nodes
- **Default**: First row pre-selected
- **Node clicks**: Not supported
