## Data Lake Ingestion & Indexing
- After data lake ingestion and indexing, display to users:
  - Join paths that were extracted
  - Semantic types discovered by semantic annotation

## Query Side UI Improvements

### Join Path Visualization
- Display matching join paths alongside the query table (side by side)
- Allow users to compare join paths
- Highlight nodes that have overlap with user-provided columns
- Highlight join paths that contain full tuples from the query table

### Interactive Highlighting
- **Row selection**: When user selects a row from the query table:
  - Selected row becomes blue
  - Highlight corresponding join paths that contain the full row
- **Column selection**: When user selects a column:
  - Highlight nodes that overlap with the column values

## Heatmap Visualization

### Chart Design
- **X-axis**: Number of nodes
- **Color intensity**: Estimated number of tuples (darker = more tuples)
- **Hover tooltip**: Display:
  - Number of columns with overlap with query column
  - Number of tuples contained
  - Other relevant metadata
- **Click interaction**: Clicking a square shows the corresponding join path

### Benefits
- Avoids overwhelming users with a long list of join paths
- Provides toggle between full list view and heatmap view
- Helps users understand whether more nodes typically correlate with better results

## Implementation Location
- Show this heatmap in two locations:
  - **Offline part**: Indexed join path section where join paths are extracted
  - **Query results**: Final retrieved top K join paths
- Both screens should include a toggle between full list view and heatmap view

