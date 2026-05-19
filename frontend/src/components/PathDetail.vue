<script setup>
import JoinGraph from './JoinGraph.vue'

defineProps({
  path: Object,
  busy: Boolean,
  highlightColumn: {
    type: String,
    default: null
  },
  tableColumns: {
    type: Object,
    default: () => ({})
  }
})

const emit = defineEmits(['materialize'])
</script>

<template>
  <section v-if="path" class="path-detail">
    <div class="section-heading">
      <div>
        <h2>{{ path.nodes.join(' — ') }}</h2>
        <p>{{ path.tupleCount }} rows · {{ path.columnCount }} columns · score {{ path.score.toFixed(3) }}</p>
      </div>
      <button class="btn btn-primary" type="button" :disabled="busy" @click="emit('materialize', path)">
        Materialize
      </button>
    </div>
    <JoinGraph
      :path="path"
      title="Path Graph"
      :highlight-column="highlightColumn"
      :table-columns="tableColumns"
    />
    <div class="attribute-strip">
      <span v-for="column in path.columns" :key="column">{{ column }}</span>
    </div>
  </section>
</template>
