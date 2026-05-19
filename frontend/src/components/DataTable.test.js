import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import DataTable from './DataTable.vue'

describe('DataTable', () => {
  it('emits a selected column when a selectable header is clicked', async () => {
    const wrapper = mount(DataTable, {
      props: {
        selectableColumns: true,
        selectedColumn: 'country',
        data: {
          columns: ['order_id', 'country'],
          rows: [{ order_id: 'O100', country: 'US' }]
        }
      }
    })

    await wrapper.findAll('.column-header-btn')[0].trigger('click')

    expect(wrapper.emitted('select-column')[0]).toEqual(['order_id'])
    expect(wrapper.findAll('th')[1].classes()).toContain('column-selected')
  })

  it('emits and marks a selected row when rows are selectable', async () => {
    const wrapper = mount(DataTable, {
      props: {
        selectableRows: true,
        selectedRowIndex: 1,
        data: {
          columns: ['order_id', 'country'],
          rows: [
            { order_id: 'O100', country: 'US' },
            { order_id: 'O101', country: 'SG' }
          ]
        }
      }
    })

    await wrapper.findAll('tbody tr')[0].trigger('click')

    expect(wrapper.emitted('select-row')[0]).toEqual([0])
    expect(wrapper.findAll('tbody tr')[1].classes()).toContain('row-selected')
  })
})
