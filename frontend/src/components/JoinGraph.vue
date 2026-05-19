<script setup>
import { computed } from 'vue'

const props = defineProps({
  graph: Object,
  path: Object,
  title: {
    type: String,
    default: 'Join Graph'
  },
  defaultOpen: {
    type: Boolean,
    default: false
  },
  highlightColumn: {
    type: String,
    default: null
  },
  tableColumns: {
    type: Object,
    default: () => ({})
  }
})

const nodes = computed(() => {
  if (props.path) {
    return props.path.nodes.map((id) => ({
      id,
      label: id === 'query' ? 'Query' : id,
      columns: props.tableColumns?.[id] || []
    }))
  }
  return (props.graph?.nodes || []).map((node) => ({
    ...node,
    columns: node.columns || props.tableColumns?.[node.id] || []
  }))
})

const edges = computed(() => {
  if (props.path) return props.path.edges || []
  return props.graph?.edges || []
})

const edgeCounts = computed(() => {
  return edges.value.reduce(
    (counts, edge) => {
      counts[edgeType(edge)] = (counts[edgeType(edge)] || 0) + 1
      return counts
    },
    { equi: 0, semantic: 0 }
  )
})

const width = computed(() => Math.max(640, nodes.value.length * 150))
const height = computed(() => Math.max(320, nodes.value.length * 72))

// Deterministic PRNG so the "random" scatter is stable for a given set of
// nodes — it should look organic, not reshuffle on every re-render.
function hashSeed(str) {
  let h = 2166136261
  for (let i = 0; i < str.length; i += 1) {
    h ^= str.charCodeAt(i)
    h = Math.imul(h, 16777619)
  }
  return h >>> 0
}

function mulberry32(seed) {
  let a = seed >>> 0
  return function next() {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

const positions = computed(() => {
  const pos = new Map()
  const list = nodes.value
  const w = width.value
  const h = height.value
  const marginX = 72
  const marginY = 50
  const rand = mulberry32(
    hashSeed(
      list
        .map((n) => n.id)
        .slice()
        .sort()
        .join('|')
    )
  )
  const placed = []
  list.forEach((node) => {
    let x = w / 2
    let y = h / 2
    for (let attempt = 0; attempt < 28; attempt += 1) {
      x = marginX + rand() * (w - marginX * 2)
      y = marginY + rand() * (h - marginY * 2)
      const clash = placed.some((p) => Math.abs(p.x - x) < 116 && Math.abs(p.y - y) < 54)
      if (!clash) break
    }
    placed.push({ x, y })
    pos.set(node.id, { x, y })
  })
  return pos
})

// Every edge is drawn as a quadratic curve. Edges between the same node pair
// (e.g. an equi join and a semantic join on the same two tables) share
// endpoints, so they're fanned apart along the perpendicular — each gets its
// own bow and label instead of stacking on top of the first one drawn.
const renderedEdges = computed(() => {
  const groups = new Map()
  edges.value.forEach((edge) => {
    if (!positions.value.has(edge.left) || !positions.value.has(edge.right)) return
    const pair = [edge.left, edge.right].slice().sort().join('::')
    if (!groups.has(pair)) groups.set(pair, [])
    groups.get(pair).push(edge)
  })

  const baseBow = 30
  const spread = 42
  const out = []
  groups.forEach((groupEdges) => {
    const count = groupEdges.length
    groupEdges.forEach((edge, i) => {
      const p1 = positions.value.get(edge.left)
      const p2 = positions.value.get(edge.right)
      const dx = p2.x - p1.x
      const dy = p2.y - p1.y
      const len = Math.hypot(dx, dy) || 1
      const nx = -dy / len
      const ny = dx / len
      const mx = (p1.x + p2.x) / 2
      const my = (p1.y + p2.y) / 2
      // apex displacement of the curve from the straight chord
      const apex = baseBow + (i - (count - 1) / 2) * spread
      const cx = mx + nx * apex * 2
      const cy = my + ny * apex * 2
      out.push({
        key: `${edge.left}-${edge.right}-${edge.key}-${edge.rightKey || edge.right_key || ''}-${edgeType(edge)}-${i}`,
        d: `M ${p1.x} ${p1.y} Q ${cx} ${cy} ${p2.x} ${p2.y}`,
        labelX: mx + nx * apex,
        labelY: my + ny * apex - 4,
        color: edgeColor(edge),
        label: edgeLabel(edge)
      })
    })
  })
  return out
})

function edgeType(edge) {
  return String(edge?.joinType || edge?.join_type || 'equi').toLowerCase()
}

function edgeColor(edge) {
  return edgeType(edge) === 'semantic' ? '#f97316' : '#2563eb'
}

function edgeLabel(edge) {
  return `${edge.key} / ${edge.rightKey || edge.right_key || edge.key}`
}

function normalizedColumn(column) {
  return String(column || '').trim().toLowerCase()
}

function nodeHasHighlightColumn(node) {
  const selected = normalizedColumn(props.highlightColumn)
  if (!selected || node.id === 'query') return false
  return (node.columns || []).some((column) => normalizedColumn(column) === selected)
}

function nodeFill(node) {
  if (nodeHasHighlightColumn(node)) return '#dcfce7'
  return node.id === 'query' ? '#dbeafe' : '#f8fafc'
}

function nodeStroke(node) {
  if (nodeHasHighlightColumn(node)) return '#16a34a'
  return node.id === 'query' ? '#2563eb' : '#64748b'
}

function nodeStrokeWidth(node) {
  return nodeHasHighlightColumn(node) ? 2.75 : 1.5
}
</script>

<template>
  <details class="graph-panel" :open="defaultOpen">
    <summary>
      <span>{{ title }}</span>
      <small>
        {{ nodes.length }} nodes · {{ edges.length }} joins ·
        {{ edgeCounts.equi || 0 }} equi · {{ edgeCounts.semantic || 0 }} semantic
      </small>
    </summary>
    <div class="graph-scroll">
      <svg :viewBox="`0 0 ${width} ${height}`" :width="width" :height="height" role="img" :aria-label="title">
        <g v-for="rEdge in renderedEdges" :key="rEdge.key">
          <path :d="rEdge.d" fill="none" :stroke="rEdge.color" stroke-width="2.5" />
          <text :x="rEdge.labelX" :y="rEdge.labelY" text-anchor="middle" class="edge-label">
            {{ rEdge.label }}
          </text>
        </g>
        <g v-for="node in nodes" :key="node.id">
          <rect
            :x="positions.get(node.id).x - 48"
            :y="positions.get(node.id).y - 18"
            width="96"
            height="36"
            rx="6"
            :fill="nodeFill(node)"
            :stroke="nodeStroke(node)"
            :stroke-width="nodeStrokeWidth(node)"
          />
          <text :x="positions.get(node.id).x" :y="positions.get(node.id).y + 4" text-anchor="middle" class="node-label">
            {{ node.label || node.id }}
          </text>
        </g>
      </svg>
    </div>
    <div class="graph-legend" aria-label="Join edge types">
      <span><i class="edge-swatch edge-swatch-equi"></i>Equi-join</span>
      <span><i class="edge-swatch edge-swatch-semantic"></i>Semantic join</span>
    </div>
  </details>
</template>
