<script setup>
import { onMounted } from 'vue'
import { useRouter } from 'vue-router'
import HyperparameterControls from './HyperparameterControls.vue'
import {
  ensureDatasetReady,
  loadDatalakeOptions,
  resetWorkflow,
  selectDatalake,
  setError,
  workflow
} from '../stores/workflow'

const router = useRouter()

async function preprocess() {
  workflow.error = ''
  try {
    await ensureDatasetReady()
    router.push('/')
  } catch (error) {
    setError(error)
  }
}

function reset() {
  resetWorkflow()
  router.push('/')
}

function changeDatalake(event) {
  selectDatalake(event.target.value)
}

onMounted(async () => {
  try {
    await loadDatalakeOptions()
  } catch (error) {
    setError(error)
  }
})
</script>

<template>
  <header class="app-header">
    <div class="brand">
      <span class="brand-mark" aria-hidden="true">S</span>
      <div>
        <h1>SemDisc</h1>
        <span>Semantic join discovery</span>
      </div>
    </div>
    <div class="header-controls">
      <label class="field datalake-select">
        <span>Data lake</span>
        <select :value="workflow.selectedDatalake" @change="changeDatalake">
          <option v-for="name in workflow.datalakes" :key="name" :value="name">
            {{ name }}
          </option>
        </select>
      </label>
      <HyperparameterControls v-model="workflow.hyperparameters" />
      <button class="btn btn-primary" type="button" :disabled="workflow.isPopulating" @click="preprocess">
        {{ workflow.isPopulating ? 'Preprocessing' : 'Preprocess' }}
      </button>
      <button class="btn btn-secondary" type="button" @click="reset">Reset</button>
    </div>
  </header>
</template>
