<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import EditableQueryTable from '../components/EditableQueryTable.vue'
import { loadTestQueryTable, setError, submitCurrentQuery, workflow } from '../stores/workflow'

const router = useRouter()
const selectedTestQuery = ref('')

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

async function selectTestQueryTable(event) {
  const value = event.target.value
  if (!value) return
  try {
    await loadTestQueryTable(value)
  } catch (error) {
    setError(error)
  } finally {
    selectedTestQuery.value = ''
  }
}
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
    <label class="field query-select" for="test-query-table">
      <span>Select Test Query Table</span>
      <select id="test-query-table" v-model="selectedTestQuery" @change="selectTestQueryTable">
        <option value="" disabled hidden>Query 1</option>
        <option value="query-1">Query 1</option>
      </select>
    </label>
    <EditableQueryTable
      v-model="workflow.query"
      :selected-column="workflow.selectedQueryColumn"
      @select="selectCell"
    />
  </section>
</template>
