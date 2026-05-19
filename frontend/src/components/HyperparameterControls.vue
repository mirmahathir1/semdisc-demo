<script setup>
const model = defineModel({ required: true })
</script>

<template>
  <div class="hp-grid">
    <label>
      <span>SimHash</span>
      <input v-model.number="model.simhashSize" type="range" min="32" max="512" step="32" />
      <b>{{ model.simhashSize }}</b>
    </label>
    <label>
      <span>Join threshold</span>
      <input v-model.number="model.joinEdgeThreshold" type="range" min="0" max="1" step="0.01" />
      <b>{{ model.joinEdgeThreshold.toFixed(2) }}</b>
    </label>
    <label>
      <span>MinHash</span>
      <input v-model.number="model.numHashesMinhash" type="range" min="16" max="512" step="16" />
      <b>{{ model.numHashesMinhash }}</b>
    </label>
    <label>
      <span>Semantic sim.</span>
      <input v-model.number="model.semanticTypeSimilarityThreshold" type="range" min="0" max="1" step="0.01" />
      <b>{{ model.semanticTypeSimilarityThreshold.toFixed(2) }}</b>
    </label>
    <label>
      <span>Max path</span>
      <input v-model.number="model.simplePathMaxLength" type="range" min="1" max="10" step="1" />
      <b>{{ model.simplePathMaxLength }}</b>
    </label>
  </div>
</template>
