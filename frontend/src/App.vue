<script setup>
import AppHeader from './components/AppHeader.vue'
import StageStepper from './components/StageStepper.vue'
import { workflow } from './stores/workflow'
</script>

<template>
  <AppHeader />
  <main v-if="workflow.isPopulating" class="app-shell loading-shell" aria-live="polite" aria-busy="true">
    <div class="loading-panel">
      <div class="section-heading">
        <h2>{{ workflow.status || 'Preprocessing data lake' }}</h2>
      </div>
      <div class="indeterminate-bar" role="progressbar" aria-label="Preprocessing data lake">
        <span></span>
      </div>
    </div>
  </main>
  <main v-else-if="workflow.processedReady" class="app-shell">
    <StageStepper />
    <p v-if="workflow.error" class="alert alert-error">{{ workflow.error }}</p>
    <p v-if="workflow.status" class="alert alert-status">{{ workflow.status }}</p>
    <RouterView />
  </main>
  <main v-else-if="workflow.error" class="app-shell">
    <p class="alert alert-error">{{ workflow.error }}</p>
  </main>
</template>
