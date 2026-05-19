<script setup>
import { ref } from 'vue'

const emit = defineEmits(['files'])
const names = ref('')

function onChange(event) {
  const files = Array.from(event.target.files || [])
  names.value = files.length ? `${files.length} file${files.length === 1 ? '' : 's'}` : ''
  emit('files', files)
}
</script>

<template>
  <label class="upload-button">
    <input type="file" multiple accept=".csv,.zip" @change="onChange" />
    <span>{{ names || 'Upload CSV/ZIP' }}</span>
  </label>
</template>

