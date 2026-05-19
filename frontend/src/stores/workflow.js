import { reactive } from 'vue'
import { api } from '../api/client'

export const workflow = reactive({
  datasetId: null,
  queryId: null,
  tables: [],
  graph: { nodes: [], edges: [] },
  semanticTypes: {},
  datalakes: [],
  selectedDatalake: 'orders',
  navigationPaths: [],
  queryPaths: [],
  query: { columns: [], rows: [] },
  uploadFiles: [],
  processedReady: false,
  selectedTable: null,
  selectedPathId: null,
  selectedQueryRow: null,
  selectedQueryColumn: null,
  materialized: null,
  status: '',
  error: '',
  hyperparameters: {
    simhashSize: 32,
    numHashesMinhash: 128,
    joinEdgeThreshold: 0.1,
    semanticTypeSimilarityThreshold: 0.7,
    simplePathMaxLength: 4,
    topK: 10
  }
})

export function setError(error) {
  workflow.error = error instanceof Error ? error.message : String(error || '')
}

export async function ensureDatasetReady() {
  if (!workflow.datasetId) {
    await loadSelectedDatalake()
  }
  await preprocessCurrentDataset()
}

export async function loadDatalakeOptions() {
  const payload = await api.datalakes()
  workflow.datalakes = payload.datalakes || []
  if (!workflow.datalakes.includes(workflow.selectedDatalake)) {
    workflow.selectedDatalake = workflow.datalakes[0] || ''
  }
}

export function selectDatalake(name) {
  if (workflow.selectedDatalake === name) return
  workflow.selectedDatalake = name
  workflow.datasetId = null
  workflow.queryId = null
  workflow.tables = []
  workflow.graph = { nodes: [], edges: [] }
  workflow.semanticTypes = {}
  workflow.navigationPaths = []
  workflow.queryPaths = []
  workflow.query = { columns: [], rows: [] }
  workflow.processedReady = false
  workflow.selectedTable = null
  workflow.selectedPathId = null
  workflow.selectedQueryRow = null
  workflow.selectedQueryColumn = null
  workflow.materialized = null
}

export async function loadSelectedDatalake() {
  const created = await api.datalake(workflow.selectedDatalake)
  workflow.datasetId = created.datasetId
  workflow.queryId = null
  workflow.tables = created.tables || []
  workflow.selectedTable = workflow.tables[0]?.name || null
  workflow.navigationPaths = []
  workflow.queryPaths = []
  workflow.materialized = null
}

export async function preprocessCurrentDataset() {
  if (!workflow.datasetId) return
  workflow.processedReady = false
  workflow.status = 'Preprocessing data lake'
  const result = await api.preprocess(workflow.datasetId, workflow.hyperparameters)
  workflow.tables = result.tables || workflow.tables
  workflow.status = 'Ready'
  await refreshDatasetViews()
  workflow.processedReady = true
}

export async function loadUploadedDataset(files) {
  workflow.uploadFiles = files
  workflow.processedReady = false
  const created = await api.createDataset(files)
  workflow.datasetId = created.datasetId
  workflow.queryId = null
  workflow.tables = created.tables || []
  workflow.selectedTable = workflow.tables[0]?.name || null
  workflow.navigationPaths = []
  workflow.queryPaths = []
  workflow.materialized = null
  await preprocessCurrentDataset()
}

export async function refreshDatasetViews() {
  if (!workflow.datasetId) return
  const [graphPayload, pathsPayload, semanticPayload] = await Promise.all([
    api.graph(workflow.datasetId),
    api.datasetPaths(workflow.datasetId, {
      maxHops: workflow.hyperparameters.simplePathMaxLength,
      k: workflow.hyperparameters.topK
    }),
    api.semanticTypes(workflow.datasetId)
  ])
  workflow.graph = graphPayload.graph
  workflow.navigationPaths = pathsPayload.paths || []
  workflow.semanticTypes = semanticPayload.semanticTypes || {}
  workflow.selectedPathId = workflow.navigationPaths[0]?.id || null
}

export async function ensureQueryLoaded() {
  if (workflow.query.columns.length) return
  const payload = await api.getDemoQuery(workflow.selectedDatalake)
  workflow.query = payload.query
  workflow.selectedQueryColumn = workflow.query.columns[0] || null
}

export async function submitCurrentQuery() {
  await ensureDatasetReady()
  await ensureQueryLoaded()
  const result = await api.submitQuery(workflow.datasetId, workflow.query)
  workflow.queryId = result.queryId
  workflow.materialized = null
  await refreshQueryPaths()
}

export async function refreshQueryPaths() {
  if (!workflow.queryId) return
  const payload = await api.queryPaths(workflow.queryId, {
    maxHops: workflow.hyperparameters.simplePathMaxLength,
    k: workflow.hyperparameters.topK
  })
  workflow.queryPaths = payload.paths || []
  workflow.selectedPathId = workflow.queryPaths[0]?.id || null
}

export async function materializePath(path, { query = false } = {}) {
  const payload = {
    nodes: path.nodes,
    edges: path.edges,
    pageSize: 50
  }
  if (query && workflow.queryId) {
    payload.queryId = workflow.queryId
  } else {
    payload.datasetId = workflow.datasetId
  }
  workflow.materialized = await api.materialize(payload)
}

export function resetWorkflow() {
  workflow.datasetId = null
  workflow.queryId = null
  workflow.tables = []
  workflow.graph = { nodes: [], edges: [] }
  workflow.semanticTypes = {}
  workflow.datalakes = []
  workflow.selectedDatalake = 'orders'
  workflow.navigationPaths = []
  workflow.queryPaths = []
  workflow.processedReady = false
  workflow.materialized = null
  workflow.selectedPathId = null
  workflow.selectedQueryRow = null
  workflow.selectedQueryColumn = null
  workflow.status = ''
  workflow.error = ''
}
