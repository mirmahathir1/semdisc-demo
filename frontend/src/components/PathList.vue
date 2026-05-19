<script setup>
const props = defineProps({
  paths: {
    type: Array,
    default: () => []
  },
  selectedId: String,
  selectedQueryRow: {
    type: Number,
    default: null
  }
})

const emit = defineEmits(['select'])

function edgeCount(path, type) {
  return (path.edges || []).filter((edge) => String(edge.joinType || edge.join_type || 'equi').toLowerCase() === type).length
}

function tupleCoverageLabel(path) {
  if (!path.queryTupleCount) return ''
  const matched = path.matchedQueryTupleCount || 0
  if (path.queryTupleCoverage === 'some') return `some ${matched}/${path.queryTupleCount} query tuples`
  return ''
}

function containsSelectedQueryRow(path) {
  if (!Number.isInteger(props.selectedQueryRow)) return false
  return (path.matchedQueryRowIndices || []).includes(props.selectedQueryRow)
}
</script>

<template>
  <div class="path-list">
    <button
      v-for="path in paths"
      :key="path.id"
      class="path-row"
      :class="{
        selected: path.id === selectedId,
        'some-tuples': path.queryTupleCoverage === 'some',
        'contains-selected-query-row': containsSelectedQueryRow(path)
      }"
      type="button"
      @click="emit('select', path.id)"
    >
      <span>#{{ path.rank }}</span>
      <b>{{ path.nodes.join(' — ') }}</b>
      <small>{{ path.tupleCount }} rows · {{ path.columnCount }} cols · {{ path.hopCount }} hops</small>
      <em v-if="tupleCoverageLabel(path)" class="tuple-chip">
        {{ tupleCoverageLabel(path) }}
      </em>
      <span class="path-edge-mix">
        <i class="edge-chip edge-chip-equi">{{ edgeCount(path, 'equi') }} equi</i>
        <i class="edge-chip edge-chip-semantic">{{ edgeCount(path, 'semantic') }} semantic</i>
      </span>
    </button>
    <p v-if="!paths.length" class="empty">No paths found</p>
  </div>
</template>
