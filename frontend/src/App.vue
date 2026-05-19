<script setup>
import AppHeader from './components/AppHeader.vue'
import StageStepper from './components/StageStepper.vue'
import { workflow } from './stores/workflow'
</script>

<template>
  <AppHeader />
  <main v-if="workflow.processedReady" class="app-shell">
    <StageStepper />
    <p v-if="workflow.error" class="alert alert-error">{{ workflow.error }}</p>
    <p v-if="workflow.status" class="alert alert-status">{{ workflow.status }}</p>
    <RouterView />
  </main>
</template>
