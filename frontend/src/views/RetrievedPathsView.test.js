import { mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import RetrievedPathsView from './RetrievedPathsView.vue'
import { workflow } from '../stores/workflow'

vi.mock('../stores/workflow', async (importOriginal) => {
  const original = await importOriginal()
  return {
    ...original,
    ensureQueryLoaded: vi.fn(),
    refreshQueryPaths: vi.fn(),
    submitCurrentQuery: vi.fn(),
    materializePath: vi.fn()
  }
})

describe('RetrievedPathsView', () => {
  beforeEach(() => {
    workflow.queryId = 'qry_1'
    workflow.query = {
      columns: ['order_id'],
      rows: [{ order_id: 'O100' }, { order_id: 'O101' }]
    }
    workflow.queryPaths = []
    workflow.tables = []
    workflow.selectedQueryRow = null
    workflow.selectedQueryColumn = 'order_id'
  })

  it('toggles the selected query row off when clicked again', async () => {
    const wrapper = mount(RetrievedPathsView, {
      global: {
        stubs: {
          JoinHeatmap: true,
          MaterializedTable: true,
          PathDetail: true,
          PathList: true
        }
      }
    })

    const rows = wrapper.findAll('tbody tr')

    await rows[1].trigger('click')
    expect(workflow.selectedQueryRow).toBe(1)

    await rows[1].trigger('click')
    expect(workflow.selectedQueryRow).toBeNull()
  })
})
