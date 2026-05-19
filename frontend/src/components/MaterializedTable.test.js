import { mount } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { api } from '../api/client'
import MaterializedTable from './MaterializedTable.vue'

vi.mock('../api/client', () => ({
  api: {
    materialize: vi.fn()
  }
}))

describe('MaterializedTable', () => {
  beforeEach(() => {
    api.materialize.mockReset()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('downloads the materialized rows as escaped csv', async () => {
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {})
    const appendChild = vi.spyOn(document.body, 'appendChild')
    const wrapper = mount(MaterializedTable, {
      props: {
        result: {
          data: {
            columns: ['name', 'note'],
            rows: [{ name: 'Ada', note: 'comma, quote "x"\nline' }],
            totalRows: 1
          }
        }
      }
    })

    await wrapper.get('button').trigger('click')

    expect(click).toHaveBeenCalledOnce()
    const link = appendChild.mock.calls[0][0]
    const csv = decodeURIComponent(link.href.split(',')[1])
    expect(link.download).toBe('materialized-table.csv')
    expect(csv).toBe('"name","note"\r\n"Ada","comma, quote ""x""\nline"')
  })

  it('requests all materialized rows when the rendered table is paged', async () => {
    api.materialize.mockResolvedValueOnce({
      data: {
        columns: ['id'],
        rows: [{ id: '1' }, { id: '2' }],
        totalRows: 2
      }
    })
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {})
    const appendChild = vi.spyOn(document.body, 'appendChild')
    const source = { datasetId: 'ds_1', nodes: ['orders'], edges: [], pageSize: 50 }
    const wrapper = mount(MaterializedTable, {
      props: {
        result: {
          source,
          data: {
            columns: ['id'],
            rows: [{ id: '1' }],
            totalRows: 2
          }
        }
      }
    })

    await wrapper.get('button').trigger('click')

    expect(api.materialize).toHaveBeenCalledWith({ ...source, page: 1, pageSize: 1000 })
    expect(click).toHaveBeenCalledOnce()
    const link = appendChild.mock.calls[0][0]
    const csv = decodeURIComponent(link.href.split(',')[1])
    expect(csv).toBe('"id"\r\n"1"\r\n"2"')
  })
})
