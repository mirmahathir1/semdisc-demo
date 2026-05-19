<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import { RouterLink } from 'vue-router'
import DataTable from '../components/DataTable.vue'
import JoinGraph from '../components/JoinGraph.vue'
import JoinHeatmap from '../components/JoinHeatmap.vue'
import PathDetail from '../components/PathDetail.vue'
import PathList from '../components/PathList.vue'
import MaterializedTable from '../components/MaterializedTable.vue'
import { api } from '../api/client'
import { materializePath, setError, workflow } from '../stores/workflow'

const preview = ref(null)
const selectedIds = ref([])
const busy = ref(false)

const visiblePaths = computed(() => {
  if (!selectedIds.value.length) return workflow.navigationPaths
  return workflow.navigationPaths.filter((path) => selectedIds.value.includes(path.id))
})

const selectedPath = computed(() =>
  visiblePaths.value.find((path) => path.id === workflow.selectedPathId) || visiblePaths.value[0]
)

async function loadTable() {
  if (!workflow.datasetId || !workflow.selectedTable) return
  preview.value = await api.table(workflow.datasetId, workflow.selectedTable, 1, 50)
}

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
  const first = workflow.navigationPaths.find((path) => ids.includes(path.id))
  if (first) workflow.selectedPathId = first.id
}

onMounted(async () => {
  try {
    if (!workflow.processedReady) return
    // Default selection is driven by JoinHeatmap (its densest cell).
    await loadTable()
  } catch (error) {
    setError(error)
  }
})

watch(() => workflow.selectedTable, loadTable)
</script>

<template>
  <section class="view-grid">
    <aside class="sidebar-pane">
      <div class="section-heading">
        <h2>Data Lake</h2>
        <RouterLink class="btn btn-primary" to="/query">Input query</RouterLink>
      </div>
      <label class="field">
        <span>Table</span>
        <select v-model="workflow.selectedTable">
          <option v-for="table in workflow.tables" :key="table.name" :value="table.name">
            {{ table.name }} · {{ table.rows }} rows
          </option>
        </select>
      </label>
      <div class="semantic-list" v-if="workflow.selectedTable">
        <h3>Semantic Types</h3>
        <p v-for="item in workflow.semanticTypes[workflow.selectedTable]" :key="item.column">
          <b>{{ item.column }}</b>
          <span>{{ item.semanticType }}</span>
        </p>
      </div>
      <JoinHeatmap :paths="workflow.navigationPaths" :selected-ids="selectedIds" @select="selectHeatmap" />
    </aside>

    <section class="main-pane">
      <JoinGraph :graph="workflow.graph" />

      <div class="split">
        <section>
          <div class="section-heading">
            <h2>Join Paths</h2>
            <p>{{ visiblePaths.length }} shown</p>
          </div>
          <PathList :paths="visiblePaths" :selected-id="workflow.selectedPathId" @select="workflow.selectedPathId = $event" />
        </section>
        <section>
          <PathDetail :path="selectedPath" :busy="busy" @materialize="materialize" />
        </section>
      </div>

      <MaterializedTable :result="workflow.materialized" />

      <section v-if="preview?.data" class="preview-band">
        <div class="section-heading">
          <h2>{{ preview.table }}</h2>
          <p>{{ preview.data.totalRows }} rows</p>
        </div>
        <DataTable :data="preview.data" compact />
      </section>
    </section>
  </section>
</template>
