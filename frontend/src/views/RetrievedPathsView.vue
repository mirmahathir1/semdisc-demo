<script setup>
import { computed, onMounted, ref } from 'vue'
import DataTable from '../components/DataTable.vue'
import JoinHeatmap from '../components/JoinHeatmap.vue'
import MaterializedTable from '../components/MaterializedTable.vue'
import PathDetail from '../components/PathDetail.vue'
import PathList from '../components/PathList.vue'
import {
  ensureQueryLoaded,
  materializePath,
  refreshQueryPaths,
  setError,
  submitCurrentQuery,
  workflow
} from '../stores/workflow'

const selectedIds = ref([])
const busy = ref(false)

const queryPreview = computed(() => workflow.query)
const visiblePaths = computed(() => {
  if (!selectedIds.value.length) return workflow.queryPaths
  return workflow.queryPaths.filter((path) => selectedIds.value.includes(path.id))
})
const selectedPath = computed(() =>
  visiblePaths.value.find((path) => path.id === workflow.selectedPathId) || visiblePaths.value[0]
)
const tableColumns = computed(() => {
  const columns = {
    query: workflow.query.columns || []
  }
  workflow.tables.forEach((table) => {
    columns[table.name] = table.columns || []
  })
  return columns
})

async function materialize(path) {
  busy.value = true
  try {
    await materializePath(path)
  } catch (error) {
    setError(error)
  } finally {
    busy.value = false
  }
}

function selectHeatmap(ids) {
  selectedIds.value = ids
  const first = workflow.queryPaths.find((path) => ids.includes(path.id))
  if (first) workflow.selectedPathId = first.id
}

function selectQueryColumn(column) {
  workflow.selectedQueryColumn = column
}

function selectQueryRow(rowIndex) {
  workflow.selectedQueryRow = workflow.selectedQueryRow === rowIndex ? null : rowIndex
}

onMounted(async () => {
  try {
    await ensureQueryLoaded()
    if (!workflow.queryId) {
      await submitCurrentQuery()
    } else {
      await refreshQueryPaths()
    }
    // Default selection is driven by JoinHeatmap (its densest cell).
  } catch (error) {
    setError(error)
  }
})
</script>

<template>
  <section class="retrieved-layout">
    <aside class="sidebar-pane">
      <div class="section-heading">
        <h2>Query</h2>
        <p v-if="Number.isInteger(workflow.selectedQueryRow)">Row {{ workflow.selectedQueryRow + 1 }}</p>
      </div>
      <DataTable
        :data="queryPreview"
        compact
        selectable-columns
        selectable-rows
        :selected-column="workflow.selectedQueryColumn"
        :selected-row-index="workflow.selectedQueryRow"
        @select-column="selectQueryColumn"
        @select-row="selectQueryRow"
      />
      <JoinHeatmap :paths="workflow.queryPaths" :selected-ids="selectedIds" @select="selectHeatmap" />
    </aside>

    <section class="main-pane">
      <div class="section-heading">
        <h2>Retrieved Join Paths</h2>
        <p>{{ visiblePaths.length }} shown</p>
      </div>
      <div class="retrieved-legend" aria-label="Retrieved path legend">
        <span><i class="edge-swatch edge-swatch-equi"></i>Equi-join</span>
        <span><i class="edge-swatch edge-swatch-semantic"></i>Semantic join</span>
        <span><i class="legend-chip legend-chip-all"></i>All query tuples</span>
        <span><i class="legend-chip legend-chip-some"></i>Some query tuples</span>
        <span><i class="legend-node legend-node-query"></i>Query-column table</span>
        <span><i class="legend-node legend-node-selected"></i>Selected-column table</span>
      </div>
      <div class="split">
        <PathList
          :paths="visiblePaths"
          :selected-id="workflow.selectedPathId"
          :selected-query-row="workflow.selectedQueryRow"
          @select="workflow.selectedPathId = $event"
        />
        <PathDetail
          :path="selectedPath"
          :busy="busy"
          :graph-edges="workflow.graph.edges"
          :highlight-column="workflow.selectedQueryColumn"
          :query-columns="workflow.query.columns"
          :table-columns="tableColumns"
          @materialize="materialize"
        />
      </div>
      <MaterializedTable :result="workflow.materialized" />
    </section>
  </section>
</template>
