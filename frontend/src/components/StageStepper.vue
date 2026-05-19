<script setup>
import { computed } from 'vue'
import { useRoute, RouterLink } from 'vue-router'

const route = useRoute()
const stages = [
  { path: '/', label: 'Navigate' },
  { path: '/query', label: 'Input query' },
  { path: '/retrieved', label: 'Retrieved paths' }
]

const activeIndex = computed(() => Math.max(0, stages.findIndex((stage) => stage.path === route.path)))
</script>

<template>
  <nav class="stage-stepper" aria-label="Workflow stages">
    <div class="stage-line">
      <span :style="{ width: `${((activeIndex + 1) / stages.length) * 100}%` }"></span>
    </div>
    <RouterLink
      v-for="(stage, index) in stages"
      :key="stage.path"
      :to="stage.path"
      class="stage"
      :class="{ active: index === activeIndex, done: index < activeIndex }"
    >
      <span class="stage-dot"></span>
      <span>{{ stage.label }}</span>
    </RouterLink>
  </nav>
</template>

