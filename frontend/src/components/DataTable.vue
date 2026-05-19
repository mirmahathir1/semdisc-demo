<script setup>
const props = defineProps({
  data: {
    type: Object,
    default: () => ({ columns: [], rows: [] })
  },
  compact: Boolean,
  selectableColumns: Boolean,
  selectableRows: Boolean,
  selectedRowIndex: {
    type: Number,
    default: null
  },
  selectedColumn: {
    type: String,
    default: null
  }
})

const emit = defineEmits(['select-column', 'select-row'])

function selectColumn(column) {
  if (!props.selectableColumns) return
  emit('select-column', column)
}

function selectRow(rowIndex) {
  if (!props.selectableRows) return
  emit('select-row', rowIndex)
}
</script>

<template>
  <div class="table-wrap" :class="{ compact }">
    <table>
      <thead>
        <tr>
          <th
            v-for="column in data.columns"
            :key="column"
            :class="{
              'column-selectable': selectableColumns,
              'column-selected': selectableColumns && column === selectedColumn
            }"
          >
            <button
              v-if="selectableColumns"
              class="column-header-btn"
              type="button"
              :aria-pressed="column === selectedColumn"
              @click="selectColumn(column)"
            >
              {{ column }}
            </button>
            <template v-else>{{ column }}</template>
          </th>
        </tr>
      </thead>
      <tbody>
        <tr
          v-for="(row, rowIndex) in data.rows"
          :key="rowIndex"
          :class="{
            'row-selectable': selectableRows,
            'row-selected': selectableRows && rowIndex === selectedRowIndex
          }"
          :tabindex="selectableRows ? 0 : undefined"
          :role="selectableRows ? 'button' : undefined"
          @click="selectRow(rowIndex)"
          @keydown.enter.prevent="selectRow(rowIndex)"
          @keydown.space.prevent="selectRow(rowIndex)"
        >
          <td v-for="column in data.columns" :key="column">{{ row[column] ?? '' }}</td>
        </tr>
      </tbody>
    </table>
    <p v-if="!data.rows?.length" class="empty">No rows</p>
  </div>
</template>
