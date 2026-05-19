<script setup>
import DataTable from './DataTable.vue'

defineProps({
  result: Object
})
</script>

<template>
  <section v-if="result?.data" class="materialized">
    <div class="section-heading">
      <h2>Materialized Table</h2>
      <p>{{ result.data.totalRows }} rows</p>
    </div>
    <DataTable :data="result.data" />
  </section>
</template>

