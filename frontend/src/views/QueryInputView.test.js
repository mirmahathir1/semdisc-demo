import { mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import QueryInputView from './QueryInputView.vue'
import { loadTestQueryTable, workflow } from '../stores/workflow'

vi.mock('vue-router', () => ({
  useRouter: () => ({
    push: vi.fn()
  })
}))

vi.mock('../stores/workflow', async (importOriginal) => {
  const original = await importOriginal()
  return {
    ...original,
    loadTestQueryTable: vi.fn(async (value) => {
      if (value !== 'query-1') return
      original.workflow.query = {
        columns: ['order_id'],
        rows: [{ order_id: 'O100' }]
      }
    }),
    submitCurrentQuery: vi.fn(),
    setError: vi.fn()
  }
})

describe('QueryInputView', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    workflow.query = { columns: [], rows: [] }
    workflow.selectedQueryColumn = null
    workflow.selectedQueryRow = null
  })

  it('starts blank and loads Query 1 only when selected', async () => {
    const wrapper = mount(QueryInputView)

    expect(wrapper.text()).toContain('Select Test Query Table')
    expect(wrapper.findAll('.semantic-th-input')).toHaveLength(0)

    const select = wrapper.get('#test-query-table')
    await select.setValue('query-1')

    expect(loadTestQueryTable).toHaveBeenCalledWith('query-1')
    expect(wrapper.findAll('.semantic-th-input')).toHaveLength(1)
    expect(select.element.value).toBe('')
  })
})
