const API_BASE = import.meta.env.VITE_API_BASE || '/api'

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: options.body instanceof FormData ? {} : { 'Content-Type': 'application/json' },
    ...options
  })
  const payload = await response.json().catch(() => ({}))
  if (!response.ok) {
    const message = payload?.error?.message || `Request failed with ${response.status}`
    throw new Error(message)
  }
  return payload
}

export const api = {
  health: () => request('/health'),
  getDemoDatalake: () => request('/demo/datalake'),
  datalakes: () => request('/datalakes'),
  datalake: (name) => request(`/datalakes/${encodeURIComponent(name)}`),
  getDemoQuery: (datalake) => request(`/demo/query${datalake ? `?datalake=${encodeURIComponent(datalake)}` : ''}`),
  createDataset(files) {
    const form = new FormData()
    files.forEach((file) => form.append('files', file))
    return request('/datasets', { method: 'POST', body: form })
  },
  preprocess(datasetId, hyperparameters) {
    return request('/preprocess', {
      method: 'POST',
      body: JSON.stringify({ datasetId, hyperparameters })
    })
  },
  tables: (datasetId) => request(`/datasets/${datasetId}/tables`),
  table: (datasetId, tableName, page = 1, pageSize = 100) =>
    request(`/datasets/${datasetId}/tables/${encodeURIComponent(tableName)}?page=${page}&pageSize=${pageSize}`),
  semanticTypes: (datasetId) => request(`/datasets/${datasetId}/semantic-types`),
  graph: (datasetId) => request(`/datasets/${datasetId}/graph`),
  datasetPaths: (datasetId, { maxHops = 4, k = 10 } = {}) =>
    request(`/datasets/${datasetId}/paths?maxHops=${maxHops}&k=${k}`),
  submitQuery(datasetId, query) {
    return request('/query', {
      method: 'POST',
      body: JSON.stringify({ datasetId, query })
    })
  },
  queryPaths: (queryId, { maxHops = 4, k = 10 } = {}) =>
    request(`/query/${queryId}/paths?maxHops=${maxHops}&k=${k}`),
  materialize(payload) {
    return request('/materialize', {
      method: 'POST',
      body: JSON.stringify(payload)
    })
  }
}
