<script setup>
import { computed, ref } from 'vue'
import { api } from '../api/client'
import DataTable from './DataTable.vue'

const props = defineProps({
  result: Object
})

const isDownloadingCsv = ref(false)

function csvValue(value) {
  if (value == null) return ''
  return `"${String(value).replaceAll('"', '""')}"`
}

function dataToCsv(data) {
  const columns = data?.columns || []
  if (!columns.length) return ''
  const rows = data.rows || []
  return [
    columns.map(csvValue).join(','),
    ...rows.map((row) => columns.map((column) => csvValue(row[column])).join(','))
  ].join('\r\n')
}

const csvContent = computed(() => dataToCsv(props.result?.data))

const canDownloadCsv = computed(() => Boolean(csvContent.value) && !isDownloadingCsv.value)

async function csvData() {
  const data = props.result?.data
  const source = props.result?.source
  if (!source || !data?.totalRows || (data.rows?.length || 0) >= data.totalRows) return data

  const pageSize = 1000
  const totalPages = Math.max(1, Math.ceil(data.totalRows / pageSize))
  const rows = []
  for (let page = 1; page <= totalPages; page += 1) {
    const payload = await api.materialize({ ...source, page, pageSize })
    rows.push(...(payload.data?.rows || []))
  }
  return { ...data, rows, page: 1, pageSize, totalPages }
}

async function downloadCsv() {
  if (!csvContent.value || isDownloadingCsv.value) return
  isDownloadingCsv.value = true
  try {
    const csv = dataToCsv(await csvData())
    const link = document.createElement('a')
    link.href = `data:text/csv;charset=utf-8,${encodeURIComponent(csv)}`
    link.download = 'materialized-table.csv'
    document.body.appendChild(link)
    link.click()
    link.remove()
  } finally {
    isDownloadingCsv.value = false
  }
}
</script>

<template>
  <section v-if="result?.data" class="materialized">
    <div class="section-heading">
      <div>
        <h2>Materialized Table</h2>
        <p>{{ result.data.totalRows }} rows</p>
      </div>
      <button class="btn btn-secondary" type="button" :disabled="!canDownloadCsv" @click="downloadCsv">
        {{ isDownloadingCsv ? 'Downloading...' : 'Download CSV' }}
      </button>
    </div>
    <DataTable :data="result.data" />
  </section>
</template>
