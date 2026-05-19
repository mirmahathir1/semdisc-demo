<script setup>
import { onMounted } from 'vue'
import { useRouter } from 'vue-router'
import EditableQueryTable from '../components/EditableQueryTable.vue'
import { ensureQueryLoaded, setError, submitCurrentQuery, workflow } from '../stores/workflow'

const router = useRouter()

async function submit() {
  try {
    await submitCurrentQuery()
    router.push('/retrieved')
  } catch (error) {
    setError(error)
  }
}

function selectCell({ rowIndex, column }) {
  if (Number.isInteger(rowIndex)) workflow.selectedQueryRow = rowIndex
  workflow.selectedQueryColumn = column
}

onMounted(async () => {
  try {
    if (!workflow.processedReady) return
    await ensureQueryLoaded()
  } catch (error) {
    setError(error)
  }
})
</script>

<template>
  <section class="query-editor">
    <div class="section-heading">
      <div>
        <h2>Query Table</h2>
        <p>{{ workflow.query.rows.length }} rows · {{ workflow.query.columns.length }} semantic columns</p>
      </div>
      <button class="btn btn-primary" type="button" @click="submit">Submit query table</button>
    </div>
    <p class="hint">Column headers are semantic types — edit a header to rename its semantic type.</p>
    <EditableQueryTable
      v-model="workflow.query"
      :selected-column="workflow.selectedQueryColumn"
      @select="selectCell"
    />
  </section>
</template>
