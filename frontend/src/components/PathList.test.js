import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import PathList from './PathList.vue'

describe('PathList', () => {
  it('marks paths with complete query tuple coverage in blue and partial coverage in yellow', () => {
    const wrapper = mount(PathList, {
      props: {
        paths: [
          {
            id: 'p1',
            rank: 1,
            nodes: ['orders', 'customers'],
            tupleCount: 4,
            columnCount: 8,
            hopCount: 1,
            edges: [],
            queryTupleCoverage: 'all',
            matchedQueryTupleCount: 4,
            queryTupleCount: 4
          },
          {
            id: 'p2',
            rank: 2,
            nodes: ['shipments', 'orders', 'customers'],
            tupleCount: 3,
            columnCount: 10,
            hopCount: 2,
            edges: [],
            queryTupleCoverage: 'some',
            matchedQueryTupleCount: 3,
            queryTupleCount: 4
          }
        ]
      }
    })

    const rows = wrapper.findAll('.path-row')

    expect(rows[0].classes()).toContain('all-tuples')
    expect(rows[0].classes()).not.toContain('some-tuples')
    expect(rows[0].text()).toContain('all 4/4 query tuples')
    expect(rows[1].classes()).toContain('some-tuples')
    expect(rows[1].text()).toContain('some 3/4 query tuples')
  })

  it('marks paths containing the selected query row in green', () => {
    const wrapper = mount(PathList, {
      props: {
        selectedQueryRow: 3,
        paths: [
          {
            id: 'p1',
            rank: 1,
            nodes: ['orders', 'customers'],
            tupleCount: 4,
            columnCount: 8,
            hopCount: 1,
            edges: [],
            matchedQueryRowIndices: [0, 1, 2, 3]
          },
          {
            id: 'p2',
            rank: 2,
            nodes: ['shipments', 'orders', 'customers'],
            tupleCount: 3,
            columnCount: 10,
            hopCount: 2,
            edges: [],
            matchedQueryRowIndices: [0, 1, 2]
          }
        ]
      }
    })

    const rows = wrapper.findAll('.path-row')

    expect(rows[0].classes()).toContain('contains-selected-query-row')
    expect(rows[1].classes()).not.toContain('contains-selected-query-row')
  })
})
