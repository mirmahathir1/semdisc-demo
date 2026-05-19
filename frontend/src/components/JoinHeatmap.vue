<script setup>
import { computed, watch } from 'vue'

const props = defineProps({
  paths: {
    type: Array,
    default: () => []
  },
  selectedIds: {
    type: Array,
    default: () => []
  }
})

const emit = defineEmits(['select'])

const BIN_COUNT = 10

const nodeCounts = computed(() => {
  return Array.from(new Set(props.paths.map((path) => Number(path.nodeCount || path.nodes?.length || 0))))
    .filter((count) => count > 0)
    .sort((a, b) => a - b)
})

const bins = computed(() => {
  if (!props.paths.length) return []

  const counts = props.paths.map((path) => Number(path.tupleCount || 0))
  const min = Math.min(...counts)
  const max = Math.max(...counts)

  if (min === max) {
    return [{ index: 0, lo: min, hi: max, label: formatBin(min, max) }]
  }

  const step = Math.max(1, Math.ceil((max - min + 1) / BIN_COUNT))
  const result = []
  let index = 0
  for (let lo = min; lo <= max; lo += step) {
    const hi = Math.min(max, lo + step - 1)
    result.push({ index, lo, hi, label: formatBin(lo, hi) })
    index += 1
  }
  return result
})

const rows = computed(() => {
  const grouped = new Map()

  props.paths.forEach((path) => {
    const tupleCount = Number(path.tupleCount || 0)
    const nodeCount = Number(path.nodeCount || path.nodes?.length || 0)
    const bin = bins.value.find((candidate) => tupleCount >= candidate.lo && tupleCount <= candidate.hi)
    if (!bin || !nodeCount) return

    const key = `${bin.index}:${nodeCount}`
    if (!grouped.has(key)) grouped.set(key, [])
    grouped.get(key).push(path)
  })

  return bins.value.map((bin) => {
    return {
      bin,
      cells: nodeCounts.value.map((nodeCount) => {
        const key = `${bin.index}:${nodeCount}`
        return {
          key,
          nodeCount,
          bin,
          paths: grouped.get(key) || []
        }
      })
    }
  })
})

const maxCellCount = computed(() => {
  return Math.max(1, ...rows.value.flatMap((row) => row.cells.map((cell) => cell.paths.length)))
})

// The single cell with the most join paths (the darkest cell). Ties resolve
// to the first one encountered (lowest tuple bin, then fewest tables).
const densestCellIds = computed(() => {
  let best = null
  rows.value.forEach((row) => {
    row.cells.forEach((cell) => {
      if (cell.paths.length && (!best || cell.paths.length > best.paths.length)) {
        best = cell
      }
    })
  })
  return best ? best.paths.map((path) => path.id) : []
})

// Stable signature of the underlying path set so we only reset the default
// selection when the data actually changes, not on every re-render.
const pathSignature = computed(() => props.paths.map((path) => path.id).join('|'))

watch(
  pathSignature,
  () => {
    const ids = densestCellIds.value
    if (ids.length) emit('select', ids)
  },
  { immediate: true }
)

function formatBin(lo, hi) {
  return lo === hi ? `${lo}` : `${lo}-${hi}`
}

function selectCell(cell) {
  if (!cell.paths.length) return
  emit('select', cell.paths.map((path) => path.id))
}
</script>

<template>
  <div v-if="paths.length" class="heatmap-wrap">
    <div class="heatmap-yaxis">Tuple count</div>
    <div class="heatmap-main">
      <div class="heatmap-xaxis">Table count</div>
      <div class="heatmap-grid" :style="{ '--node-columns': nodeCounts.length }" role="grid">
        <div class="heatmap-corner" role="presentation"></div>
        <div v-for="nodeCount in nodeCounts" :key="nodeCount" class="heatmap-header" role="columnheader">
          {{ nodeCount }}
        </div>

        <template v-for="row in rows" :key="row.bin.index">
          <div class="heatmap-bin" role="rowheader">{{ row.bin.label }}</div>
          <button
            v-for="cell in row.cells"
            :key="cell.key"
            type="button"
            class="heatmap-cell"
            :class="{ selected: cell.paths.some((path) => selectedIds.includes(path.id)) }"
            :disabled="!cell.paths.length"
            :style="{ '--intensity': cell.paths.length / maxCellCount }"
            :title="`${cell.paths.length} paths with ${cell.nodeCount} tables and ${cell.bin.label} tuples`"
            @click="selectCell(cell)"
          ></button>
        </template>
      </div>
    </div>
  </div>
  <p v-else class="empty">No paths found</p>
</template>
