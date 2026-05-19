<script setup>
const model = defineModel({ required: true })
const props = defineProps({
  selectedColumn: {
    type: String,
    default: null
  }
})
const emit = defineEmits(['select'])

function addRow() {
  const row = Object.fromEntries(model.value.columns.map((column) => [column, '']))
  model.value.rows.push(row)
}

function removeRow(index) {
  model.value.rows.splice(index, 1)
}

function addColumn() {
  const name = `semantic_${model.value.columns.length + 1}`
  model.value.columns.push(name)
  model.value.rows.forEach((row) => {
    row[name] = ''
  })
}

function renameColumn(index, nextName) {
  const oldName = model.value.columns[index]
  const clean = nextName.trim()
  if (!clean || clean === oldName || model.value.columns.includes(clean)) return
  model.value.columns[index] = clean
  model.value.rows.forEach((row) => {
    row[clean] = row[oldName]
    delete row[oldName]
  })
  if (props.selectedColumn === oldName) selectColumn(clean)
}

function selectColumn(column) {
  emit('select', { rowIndex: null, column })
}

function selectCell(rowIndex, column) {
  emit('select', { rowIndex, column })
}
</script>

<template>
  <div class="editable-table">
    <div class="table-actions">
      <button class="btn btn-secondary" type="button" @click="addColumn">Add column</button>
      <button class="btn btn-secondary" type="button" @click="addRow">Add row</button>
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th
              v-for="(column, index) in model.columns"
              :key="index"
              class="semantic-th"
              :class="{ 'column-selected': column === selectedColumn }"
            >
              <input
                class="semantic-th-input"
                :value="column"
                type="text"
                aria-label="Semantic type"
                title="Semantic type — edit to rename this column"
                @focus="selectColumn(column)"
                @click="selectColumn(column)"
                @change="renameColumn(index, $event.target.value)"
              />
            </th>
            <th class="thin"></th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(row, rowIndex) in model.rows" :key="rowIndex">
            <td v-for="column in model.columns" :key="column">
              <input
                v-model="row[column]"
                type="text"
                @focus="selectCell(rowIndex, column)"
              />
            </td>
            <td class="thin">
              <button class="icon-btn" type="button" title="Remove row" @click="removeRow(rowIndex)">x</button>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>
